window.onload = function(){
    
    document.getElementById("usuario").oninput =guardarUsuario
    
    };
    
    
    
    function guardarUsuario(){
        if(document.getElementById("usuario").value != "" )
        {           
            localStorage.setItem('usuarioLogInCorreo',  document.getElementById("usuario").value);  
            document.getElementById("continuar").enabled="true"  ;       
        }
    }