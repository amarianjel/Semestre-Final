import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

const rootElement = document.getElementById("root");

class Clock extends React.Component {
  constructor(props) {
    super(props);
    this.state = {date:new Date()};
  }


  componentDidMount() {
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick(){
    this.setState({
      date:new Date()
    });
  }

  render() {
    return (
      <div>
        <h1> Holaaaa, Mundo!!!</h1>
        <h2> Son las {this.state.date.toLocaleTimeString()}.</h2>
      </div>
    );
  }
}


// Mis Funciones

class Ciclo extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      nombre: "Abraham Marianjel",
      link: "http://randomcolour.com/"
    }; 
  }

  componentDidMount() {
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick(){
    this.setState({
      link:"http://randomcolour.com/"
    });
  }

  render() {
    return (
      <div>
        <h1> Hola {this.state.nombre}</h1>
        <iframe src={this.state.link} className="colores"></iframe>
      </div>
    );
  }
}

ReactDOM.render(
  <Ciclo/>,
  rootElement
);