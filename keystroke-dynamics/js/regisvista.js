

window.onload = function(){
    
document.getElementById("correo").oninput =guardarUsuario

};



function guardarUsuario(){
    if(document.getElementById("correo").value != "" && document.getElementById("nombre").value != ""  )
    {
        localStorage.setItem('usuarioNuevoNombre',  document.getElementById("nombre").value);
        localStorage.setItem('usuarioNuevoCorreo',  document.getElementById("correo").value);
        document.getElementById("text").hidden = false;
        document.getElementById("registro").hidden = false;
    }
}