import { useEffect, useRef, useState } from "react";
import { Post } from "./";
import { v4 as uuid } from "uuid";

const key = 'notas';

export function List(){

    const [ notas, setNotas ] = useState([
        { id:1,titulo:"Subir Notas",descripcion:"Estudiar para sacar buen promedio de calificaciones",important:false },
        { id:2,titulo:"Jugar Voley",descripcion:"Entrenar voleibol de jueves a viernes",important:true },
        { id:3,titulo:"Tesis",descripcion:"Trabajar en la tesis para sacar la carrera pronto",important:false },
        { id:4,titulo:"Investigar IA",descripcion:"Investigar sobre la IA y sus ventajas",important:true }
    ]);

    const tituloRef = useRef();
    const descripcionRef =useRef();
    const impotantRef =useRef();

    useEffect( () => {
        const storedPostit = JSON.parse( localStorage.getItem(key) );
        if(storedPostit){
            setNotas(storedPostit);
        }
    },[])

    useEffect( () => {
        localStorage.setItem( key,JSON.stringify(notas) );
    },[notas])
    

    const agregarNotas= () => {
       
        let titulo = tituloRef.current.value;
        const descripcion = descripcionRef.current.value;
        const important = impotantRef.current.checked;
       
        
        if(titulo === ""){
            titulo = "Sin Título";
        }

        if(descripcion === ""){
            return;
        }
        
        setNotas( ( prevNotas ) => {
            const newNotas = {
                id:uuid(),
                titulo:titulo,
                descripcion:descripcion,
                important:important
            }
            
            return [ ...prevNotas , newNotas ];
        })
        tituloRef.current.value = null;
        descripcionRef.current.value = null;
        impotantRef.current.checked = null;
    }

    const eliminarNotas = (id) => {
        
        // eslint-disable-next-line eqeqeq
        const newNotas = notas.filter( (nota) => nota.id != id);
        setNotas(newNotas);
    }

    return(
        <>
            <div className="row g-3">
                <div className="col-3">
                    <input type="text" className="form-control" ref={ tituloRef } placeholder="Titulo"  />
                </div>
                <div className="col-4">
                    <input type="text" className="form-control" ref={ descripcionRef } placeholder="Descripcion"/>
                </div>
                <div className="col-auto d-flex align-items-center">
                    <input className="form-check-input" type="checkbox" ref={ impotantRef } /><label className="white ms-1">Importante!</label>
                </div>
                <div className="col-auto d-flex align-items-center">
                    <button onClick={ agregarNotas } className="btn btn-dark boton" data-bs-toggle="button">AGREGAR</button>
                </div>
            </div>
            <br />
            <ul>
                { notas.map( ( nota )=>(
                    <Post postit={ nota } key={ nota.id } eliminar={ eliminarNotas }></Post>
                ))}
            </ul>
        </>
    );
}