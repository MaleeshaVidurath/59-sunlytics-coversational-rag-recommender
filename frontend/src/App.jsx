import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: "Hi! I'm your M2 Fashion Recommender. Tell me what you're looking for today (e.g., 'a stylish dark top').",
      articleId: null,
      imageUrl: null
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    
    // Add user message to chat
    const newMessages = [...messages, { id: Date.now(), sender: 'user', text: userQuery }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Call the FastAPI backend
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userQuery }),
      });

      if (!response.ok) {
        throw new Error('API Error');
      }

      const data = await response.json();
      
      // Update chat with bot response
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'bot',
        text: data.reply,
        articleId: data.article_id,
        imageUrl: data.image_url ? `http://localhost:8000${data.image_url}` : null
      }]);

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'bot',
        text: "Sorry, I'm having trouble connecting to the M2 Recommendation Server right now."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <Sparkles className="header-icon" size={24} />
        <h1>Sunlytics Conversational RAG</h1>
      </header>

      <main className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
            <div className="message-content">
              <div className="message-bubble">
                {msg.text}
              </div>
              
              {/* Product Image Card (if available) */}
              {msg.imageUrl && (
                <div className="product-card">
                  <img src={msg.imageUrl} alt={`Product ${msg.articleId}`} />
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-wrapper bot">
            <div className="message-bubble">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <form className="input-area" onSubmit={handleSend}>
        <input
          type="text"
          className="input-field"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me for a fashion recommendation..."
          disabled={isLoading}
        />
        <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
}

export default App;
