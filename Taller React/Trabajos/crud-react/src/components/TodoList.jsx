import React, { Fragment, useState } from 'react';

import { TodoItem } from "./TodoItem";


export function TodoList(){
    // - Lógica
    //todos: nombre de la constante a usar
    //setTodos: nombre del metodo que usará la constante
    //useState: método a usar desde el sistema
    const [todos, setTodos] = useState([
        { id:1, task:'Tarea 1' },
        { id:2, task:'Tarea 2' },
        { id:3, task:'Tarea 3' },
        { id:4, task:'Tarea 4' }, 
        { id:5, task:'Tarea 5' },
        { id:6, task:'Tarea 6' }
    ]);

    // - Vista
    return (
        <Fragment>
            <h1>Listado de Tareas</h1>
            <ul className="list-group">
                {todos.map((todo) => (
                    <TodoItem todo={ todo } key={ todo.id }/>
                ))}
            </ul>
        </Fragment>
    );
}