import React from "react";
import ReactDOM from "react-dom/client";
import './index.css';


const root = ReactDOM.createRoot(document.getElementById("root"));

// Mis funciones

function Avatar(props) {
  return (
    <img className="Avatar"
         src={props.user.avatarUrl}
         alt={props.user.name} />
  );
}

function Info(props) {
  return (
    <div className="Info">
      <div className="Info-name">
        {props.user.name}
      </div>
      <Avatar user={props.user} />
    </div>
  );
}

function Comment(props) {
  return (
    <div className="Comment row align-items-center">
      <Info user={props.author} />
      <div className="Comment-text">
        {props.text}
      </div>
    </div>
  );
}

const comment = {
  text: 'Mi foto de Facebook',
  author: {
    name: 'Abraham Marianjel',
    avatarUrl: 'https://scontent-scl2-1.xx.fbcdn.net/v/t1.18169-9/15283934_10208484356048725_3192577132137080711_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=09cbfe&_nc_ohc=XQuoc5EoGV0AX8wD_UN&_nc_ht=scontent-scl2-1.xx&oh=00_AfB1ah_As6exLd4UAD7ekhKeAXnDZlkN_6w9rnGjZUVY3A&oe=639C906B'
  }
};

//Extraigo
function App(){
  return(
  <Comment
    date={comment.date}
    text={comment.text}
    author={comment.author} />
  );
}


root.render(
  <App />,
  root
);


// function Welcome(props){
//   return <h1>Hola, {props.name}</h1>;
// }


// function App(){
//   return (
//     <div>
//       <Welcome name="Juan"/>
//       <Welcome name="Edite"/>
//     </div>
//   );
// }