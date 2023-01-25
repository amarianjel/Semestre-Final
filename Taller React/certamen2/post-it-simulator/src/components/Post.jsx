export function Post( { postit, eliminar } ){

    const { id, titulo, descripcion, important } = postit;

    const Borrar = () =>{
        eliminar(id);
    }

    return(<li>
        <span className={ important ? "importante":"normal" }>
            <div className="row">
                <h2 className="col-10">{ titulo }</h2> 
                <button className="col-2 mb-5 exis" onClick={ Borrar }>X</button>
            </div>
            <p className="d-flex align-items-start descripcion">{ descripcion }</p>
            
        </span>
        </li>
    );
}