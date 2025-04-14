import React from 'react';

function Header({ title = "Motorcyclist Accident Severity Prediction" }) {
  return (
    <div className="page-header">
      <h1 className="main-title">{title}</h1>
    </div>
  );
}

export default Header;