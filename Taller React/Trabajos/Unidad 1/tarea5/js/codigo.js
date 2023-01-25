function cambiarColor(id, color){
    el = document.getElementById(id);
    el.classList.remove('colorrojo');
    el.classList.remove('colorverde');
    el.classList.add(color);
}

function cambiarColorARojo(){
    cambiarColor('texto1', 'colorrojo');
}

function cambiarColorAVerde(){
    cambiarColor('texto1', 'colorverde');
}

function cambiarColorBanner(id, color){
    ban = document.getElementById(id);
    ban.classList.remove('colorceleste');
    ban.classList.remove('colorverde2');
    ban.classList.remove('coloramarillo');
    ban.classList.add(color);
}

function cambiarColorACeleste(){
    document.getElementById("banner-titulo").textContent="Celeste";
    cambiarColorBanner('banner', 'colorceleste');
}
function cambiarColorAVerde2(){
    document.getElementById("banner-titulo").textContent="Verde";
    cambiarColorBanner('banner', 'colorverde2');
}
function cambiarColorAAmarillo(){
    document.getElementById("banner-titulo").textContent="Amarillo";
    cambiarColorBanner('banner', 'coloramarillo');
}