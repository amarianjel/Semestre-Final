function cambiarColor(id, color){
    el = document.getElementById(id);
    el.classList.remove('facebook');
    el.classList.remove('twitter');
    el.classList.remove('instagram');
    el.classList.remove('youtube');
    el.classList.add(color);
}

function cambiarAFacebook(){
    cambiarColor('facebook', 'facebook');
}