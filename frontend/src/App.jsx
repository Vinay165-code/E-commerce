import { useEffect, useState } from 'react';
import { api } from './services/api';
import './App.css';

export default function App() {
  const [products, setProducts] = useState([]);

  useEffect(() => {
    api.get('/products')
      .then(res => {
        // FastAPI pagination returns { items: [...], total: X, ... }
        const data = res.data.items ?? res.data;
        setProducts(data);
      })
      .catch(err => console.error('API error:', err));
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold mb-6 text-center">Product Catalog</h1>
      <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {products.map(p => (
          <div key={p.id} className="border rounded-lg p-4 shadow hover:shadow-lg transition">
            {p.image_url && (
              <img
                src={p.image_url}
                alt={p.name}
                className="w-full h-48 object-cover mb-3 rounded"
              />
            )}
            <h2 className="text-xl font-semibold">{p.name}</h2>
            <p className="text-gray-700 mt-1">${p.price.toFixed(2)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
