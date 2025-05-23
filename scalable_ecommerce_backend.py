from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    status,
)
from fastapi_pagination import Page, add_pagination, paginate
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    DateTime,
    Table,
    Boolean,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
import os, json, shutil, uuid
from datetime import datetime, timedelta
import os

TESTING = os.getenv("TESTING") == "1"

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ecommerce.db")
REDIS_URL = os.getenv("REDIS_URL", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")

# ensure image directory
os.makedirs(IMAGE_DIR, exist_ok=True)

# Database setup
engine = create_engine(
    DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    ),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing & OAuth2
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Redis cache (optional)
if REDIS_URL:
    import redis

    redis_client = redis.from_url(REDIS_URL)
else:

    class DummyCache:
        def get(self, key):
            return None

        def setex(self, key, ttl, value):
            pass

        def delete(self, key):
            pass

    redis_client = DummyCache()

# Association table for categories
product_categories = Table(
    "product_categories",
    Base.metadata,
    Column("product_id", Integer, ForeignKey("products.id")),
    Column("category_id", Integer, ForeignKey("categories.id")),
)


# Models
class ProductModel(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    price = Column(Float)
    inventory = Column(Integer)
    image_url = Column(String, nullable=True)
    categories = relationship(
        "CategoryModel", secondary=product_categories, back_populates="products"
    )


class CategoryModel(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    products = relationship(
        "ProductModel", secondary=product_categories, back_populates="categories"
    )


class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="customer")
    is_verified = Column(Boolean, default=False)
    orders = relationship("OrderModel", back_populates="user")
    cart = relationship("CartModel", back_populates="user", uselist=False)


class OrderModel(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    total_amount = Column(Float)
    user = relationship("UserModel", back_populates="orders")
    items = relationship("OrderItemModel", back_populates="order")


class OrderItemModel(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer)
    price = Column(Float)
    order = relationship("OrderModel", back_populates="items")
    product = relationship("ProductModel")


class CartModel(Base):
    __tablename__ = "carts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("UserModel", back_populates="cart")
    items = relationship("CartItemModel", back_populates="cart")


class CartItemModel(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    cart_id = Column(Integer, ForeignKey("carts.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer)
    cart = relationship("CartModel", back_populates="items")
    product = relationship("ProductModel")


Base.metadata.create_all(bind=engine)


# Pydantic Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str | None = None


class ProductBase(BaseModel):
    name: str
    description: str
    price: float
    inventory: int


class ProductCreate(ProductBase):
    category_ids: list[int] = []


class Product(ProductBase):
    id: int
    image_url: str | None = None
    categories: list[str]

    class Config:
        orm_mode = True


class CategoryCreate(BaseModel):
    name: str


class Category(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str | None = "customer"


class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    is_verified: bool

    class Config:
        orm_mode = True


class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int


class OrderItem(BaseModel):
    product_id: int
    quantity: int
    price: float

    class Config:
        orm_mode = True


class Order(BaseModel):
    id: int
    user_id: int
    total_amount: float
    created_at: datetime
    items: list[OrderItem]

    class Config:
        orm_mode = True


class CartItemCreate(BaseModel):
    product_id: int
    quantity: int


class CartItem(BaseModel):
    product_id: int
    quantity: int

    class Config:
        orm_mode = True


class Cart(BaseModel):
    id: int
    user_id: int
    items: list[CartItem]

    class Config:
        orm_mode = True


# Utility functions
def send_verification_email(email: str):
    # Simulate sending an email (replace with actual email-sending logic)
    print(f"Verification email sent to {email}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user_by_email(db: Session, email: str):
    return db.query(UserModel).filter(UserModel.email == email).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    to_encode.update({"sub": data.get("sub")})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user


# Admin check dependency
def require_admin(current_user: UserModel = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


# FastAPI app init
app = FastAPI(title="Scalable E-Commerce Backend")


# --- AUTH & USER ENDPOINTS ---
@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users", response_model=User)
def create_user(
    user_in: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    user = UserModel(
        name=user_in.name,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    # Send verification email
    background_tasks.add_task(send_verification_email, user.email)
    return user


@app.get("/verify-email/{token}")
def verify_email(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=400, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")
    user = get_user_by_email(db, email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_verified = True
    db.commit()
    return {"msg": "Email verified successfully"}


@app.get("/users/me", response_model=User)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user


@app.put("/users/me", response_model=User)
def update_user_me(
    name: str = Form(None),
    email: EmailStr = Form(None),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    if name:
        current_user.name = name
    if email:
        current_user.email = email
    db.commit()
    db.refresh(current_user)
    return current_user


# --- CATEGORY ENDPOINTS ---
@app.post("/categories", response_model=Category)
def create_category(cat: CategoryCreate, db: Session = Depends(get_db)):
    if redis_client.get("categories"):
        categories = json.loads(redis_client.get("categories"))
        if any(c["name"] == cat.name for c in categories):
            raise HTTPException(status_code=400, detail="Category already exists")
    ...


@app.get("/categories", response_model=list[Category])
def list_categories(db: Session = Depends(get_db)):
    if redis_client.get("categories"):
        return json.loads(redis_client.get("categories"))
    return db.query(CategoryModel).all()


# --- PRODUCT ENDPOINTS with filtering & image upload ---
@app.post("/products/image/{product_id}")
def upload_image(
    product_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    path = os.path.join(IMAGE_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    prod = db.query(ProductModel).get(product_id)
    prod.image_url = path
    db.commit()
    return {"image_url": path}


@app.get("/products/image/{product_id}")
def serve_image(product_id: int, db: Session = Depends(get_db)):
    prod = db.query(ProductModel).get(product_id)
    if not prod or not prod.image_url:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(prod.image_url)


response_model = None if os.getenv("TESTING") == "1" else Page[Product]


@app.get("/products", response_model=None)
def list_products(
    q: str = None,
    min_price: float = None,
    max_price: float = None,
    category: str = None,
    db: Session = Depends(get_db),
):
    query = db.query(ProductModel)
    if q:
        query = query.filter(ProductModel.name.ilike(f"%{q}%"))
    if min_price is not None:
        query = query.filter(ProductModel.price >= min_price)
    if max_price is not None:
        query = query.filter(ProductModel.price <= max_price)
    if category:
        query = query.join(ProductModel.categories).filter(
            CategoryModel.name == category
        )
    products = query.all()
    return paginate(products)


@app.post("/products", dependencies=[Depends(require_admin)], response_model=None)
def create_product(item: ProductCreate, db: Session = Depends(get_db)):
    prod = ProductModel(**item.dict(exclude={"category_ids"}))
    if item.category_ids:
        prod.categories = (
            db.query(CategoryModel)
            .filter(CategoryModel.id.in_(item.category_ids))
            .all()
        )
    db.add(prod)
    db.commit()
    db.refresh(prod)
    return prod


@app.get("/products/{product_id}", response_model=None)
def get_product(product_id: int, db: Session = Depends(get_db)):
    prod = db.query(ProductModel).get(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found")
    return prod


# --- CART & ORDER ENDPOINTS ---
@app.get("/cart", response_model=Cart)
def view_cart(
    db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)
): ...
@app.post("/cart/items", response_model=Cart)
def add_to_cart(
    item: CartItemCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
): ...
@app.delete("/cart/items/{product_id}", response_model=Cart)
def remove_from_cart(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
): ...


@app.post("/orders", response_model=Order)
def order_from_cart(
    db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)
): ...


@app.get("/orders", response_model=list[Order])
def list_orders(
    db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)
):
    return db.query(OrderModel).filter(OrderModel.user_id == current_user.id).all()


add_pagination(app)

# Requirements.txt (add this to your project root):
# --------------------------------

# --------------------------------

# To run:
# 1. Install dependencies:
#    pip install fastapi uvicorn sqlalchemy pydantic[email] passlib[bcrypt] jose redis fastapi-pagination
#    # If using email validators separately:
#    pip install email-validator
# 2. Run the server:
#    uvicorn scalable_ecommerce_backend:app --reload
