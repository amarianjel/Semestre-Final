import React from 'react'
import ReactDOM from 'react-dom'
import { PostItSimulation } from './PostItSimulation';

import './styles.css'

ReactDOM.createRoot(document.getElementById('root')).render(
    <PostItSimulation/> //+ Con el <React.StrictMode> no se puede almacenar en local storage
)