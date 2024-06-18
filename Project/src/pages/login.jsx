import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './loginsignup.css';
import em from '../Assets/email.png';
import pass from '../Assets/password.png';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://localhost:3001/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data === 'exist') {
        navigate('/Profile');  // Navigate to Profile page on successful login
      } else {
        alert('Invalid email or password');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    }
  };

  return (
    <div className='container'>
      <div className='header'>
        <div className='text'>Login</div>
        <div className='underline'></div>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="input">
          <img src={em} alt="" />
          <input
            type="email"
            placeholder='Email ID'
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="input">
          <img src={pass} alt="" />
          <input
            type="password"
            placeholder='Password'
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <div className="forget-password">Lost Password? <span>Click Here!</span></div>
        <div className="submit-container">
        <button onClick={() => navigate('/Signup')} className={'gray'}>Sign Up</button>        
        <button type="submit" className={'submit'}>Login</button>
        </div>
      </form>
    </div>
  );
};

export default Login;
