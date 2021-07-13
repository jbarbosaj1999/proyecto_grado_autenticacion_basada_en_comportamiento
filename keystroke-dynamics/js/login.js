var UD = [];
var DU = [];
var DD = [];
var UU = [];
var lastdown = -1;
var lastup = -1;
var UDseq = [];
var DUseq = [];
var DDseq = [];
var UUseq = [];
var UDtest = [];
var DUtest = [];
var DDtest = [];
var UUtest = [];
var flechas = [];
var tecleo = [];
var testText = "";
var currText = "";
var reEntrenamiento = false;



function createNN(hiddenLayers, activation) {
    var options = {};
    options.hiddenLayers = hiddenLayers;
    options.activation = activation;
    nn = new ML.FNN(options);
    return nn;
  }
  
  function meanPath(data) {
    var columns = math.transpose(data);
    var meanp = columns.map(function (row) {
      return math.mean(row);
    });
    return meanp;
  }
  
  function avgAbsDevPath(data) {
    var columns = math.transpose(data);
  
    return columns.map(function (row) {
      return math.mad(row);
    });
  }
  
  function smd_nn(data, path, nnpred) {
    var avgAbsDev = avgAbsDevPath(data);
  
    return nnpred.reduce(function (sum, val, i) {
      return sum + Math.abs(path[i] - val) / avgAbsDev[i];
    }, 0);
  }
  
  function smd(data, path) {
    var avgAbsDev = avgAbsDevPath(data);
  
    var meanp = meanPath(data);
  
  
    return meanp.reduce(function (sum, val, i) {
      return sum + Math.abs(path[i] - val) / avgAbsDev[i];
    }, 0);
  }
  
  function smd_for_path(path, index, allpaths) {
    var prepath = allpaths.slice(0, index);
    var postpath = allpaths.slice(index + 1, allpaths.length);
  
  
    return smd(prepath.concat(postpath), path);
  }
  
  function maxsmd(duseq, udseq, ddseq, uuseq) {
  
    var maxdu = math.max(duseq.map(smd_for_path));
    var maxud = math.max(udseq.map(smd_for_path));
    var maxdd = math.max(ddseq.map(smd_for_path));
    var maxdd = math.max(uuseq.map(smd_for_path));
  
    return [maxdu, maxud, maxdd, maxdd];
  }
  
  function minsmd(duseq, udseq, ddseq, uuseq) {
    var mindu = math.min(duseq.map(smd_for_path));
    var minud = math.min(udseq.map(smd_for_path));
    var mindd = math.min(ddseq.map(smd_for_path));
    var minuu = math.min(uuseq.map(smd_for_path));
  
    return [mindu, minud, mindd, mindd];
  }
  
  //Same as smd above - duplicated because of refactoring can be replace with smd
  function scaledManhattanDist(x, y) {
    var columns = math.transpose(x);
    var avgAbsDev = columns.map(function (row) {
      return math.mad(row);
    });
  
    var meanp = meanPath(x);
    return meanp.reduce(function (sum, val, i) {
      return sum + Math.abs(y[i] - val) / avgAbsDev[i];
    }, 0);
  }
  

  function captureKeyEvent(e) {

    //newevent = {event :  e.type, code : e.code, keycode :  e.keyCode, keytime : e.timeStamp };
    //keyevents.push(newevent);
  
    if (e.key.length > 1) {
      if (e.key === "Backspace") {
        document.getElementById("textEntry").value="";
        currText = "";
        flechas = [];
        tecleo = [];
        DU = [];
        UD = [];
        DD = [];
        UU = [];
        return
      }
      if (e.key === "Enter" && e.type === "keyup") {
        calculardatos(tecleo);
        UD.shift();
        DD.shift();
        DU.shift();
        UU.shift();
        //console.log(DU, UD, DD, UU);
        //console.log(flechas);
        flechas = [];
        tecleo = [];
  
        if (UD.length != (DU.length - 1) || DD.length != (DU.length - 1)) {
  
  
          if (UD.length == DU.length || DD.length == DU.length) {
            if (UD.length == DU.length && DU.length == testText.length) {
  
  
              UD.shift();
  
            }
            if (DD.length == DU.length && DU.length == testText.length) {
  
              DD.shift();
  
            }
            if (UD.length == DU.length && 0 == testText.length) {
  
              UD.shift();
  
            }
            if (DD.length == DU.length && 0 == testText.length) {
  
  
              DD.shift();
  
            }
  
            else {
  
  
              alert("Se produjo un error.");
              //console.log("tipo 1")
              //console.log(UD, DU, DD, UU);

              currText = "";
              document.getElementById("textEntry").value="";
  
              flechas = [];
              DU = [];
              UD = [];
              DD = [];
              UU = [];
              lastdown = lastup = -1;
              return;
            }
          }
  
  
          else if (DD.length <= DU.length - 2 || UD.length <= DU.length - 2) {
            alert("Se produjo un error.");
            //console.log("tipo 2")
            //console.log(UD, DU, DD, UU);
            document.getElementById("textEntry").value="";

            currText = "";
            flechas = [];
            DU = [];
            UD = [];
            DD = [];
            UU = [];
            lastdown = lastup = -1;
            return;
          }
          else {
            alert("Se produjo un error.");
            //console.log("tipo 3")
            //console.log(UD, DU, DD, UU);
            document.getElementById("textEntry").value="";

            currText = "";
            flechas = [];
            DU = [];
            UD = [];
            DD = [];
            UU = [];
            lastdown = lastup = -1;
            return;
          }
  
        }
  
        if (testText === "") {
          testText = currText;
        }
  
  
  
        if (currText !== testText) {
          alert("You entered [" + currText + "], Please Enter [" + testText + "]");
          DU = [];
          UD = [];
          DD = [];
          UU = [];
          flechas = [];
          document.getElementById("textEntry").value="";
          currText = "";
          lastdown = lastup = -1;
          return;
        }
  

  
        currText = "";
  
  
  
        duavg = math.mean(DU);
        udavg = math.mean(UD);
        ddavg = math.mean(DD);
        uuavg = math.mean(UU);
  
        var test = document.getElementById("testCheck").checked;
  
        if (test) {

  
          var maxTrainError = maxsmd(DUseq, UDseq, DDseq, UUseq);
          var minTrainError = minsmd(DUseq, UDseq, DDseq, UUseq);
  
          var tvDU1 = (minTrainError[0]);
          var tvDU2 = (maxTrainError[0]);
          var trainValueDU = (tvDU2 + tvDU1) / 2;
  
  
  
          var tvUD1 = (minTrainError[1]);
          var tvUD2 = (maxTrainError[1]);
          var trainValueUD = (tvUD2 + tvUD1) / 2;
  
  
  
          var tvDD1 = minTrainError[2];
          var tvDD2 = maxTrainError[2];
          var trainValueDD = (tvDD2 + tvDD1) / 2;
  
  
  
  
          var tvUU1 = minTrainError[3];
          var tvUU2 = maxTrainError[3];
          var trainValueUU = (tvUU2 + tvUU1) / 2;  
  
  
          var stdDU = (math.std(DUseq.map(smd_for_path)) * 2.8);
  
          var stdUD = (math.std(UDseq.map(smd_for_path)) * 2.8);
  
          var stdDD = (math.std(DDseq.map(smd_for_path)) * 2.8);
  
          var stdUU = (math.std(UUseq.map(smd_for_path)) * 2.8);
  
          //console.log(Math.abs((scaledManhattanDist(DUseq, DU))) + " - " + trainValueDU + " :: " + stdDU);
          //console.log(Math.abs((scaledManhattanDist(UDseq, UD))) + " - " + trainValueUD + " :: " + stdUD);
          //console.log(Math.abs((scaledManhattanDist(DDseq, DD))) + " - " + trainValueDD + " :: " + stdDD);
          //console.log(Math.abs((scaledManhattanDist(UUseq, UU))) + " - " + trainValueUU + " :: " + stdUU);
  
          if (Math.abs((scaledManhattanDist(DUseq, DU) - trainValueDU)) <= stdDU
            && Math.abs((scaledManhattanDist(UDseq, UD) - trainValueUD)) <= stdUD
            && Math.abs((scaledManhattanDist(DDseq, DD) - trainValueDD)) <= stdDD
            && Math.abs((scaledManhattanDist(UUseq, UU) - trainValueUU)) <= stdUU) {
  
            document.getElementById("Autenticidad").innerHTML = "Es auténtico"
            document.getElementById("Autenticidad").style.visibility = "visible";
            document.getElementById("Autenticidad").style.backgroundColor = "green";
  
            if (DUseq.length <= 25) {
              DUseq.push(DU);
              UDseq.push(UD);
              DDseq.push(DD);
              UUseq.push(UU);
            }
            else if (DUseq.length > 25) {
              DUseq.shift();
              UDseq.shift();
              DDseq.shift();
              UUseq.shift();
              DUseq.push(DU);
              UDseq.push(UD);
              DDseq.push(DD);
              UUseq.push(UU);
            }  
           
            if (typeof dunn == "undefined") {
              //console.log("No Neural Network create - Failure!!!");
            }
          }
          else {
            document.getElementById("Autenticidad").innerHTML = "Es un atacante"
            document.getElementById("Autenticidad").style.visibility = "visible";
            document.getElementById("Autenticidad").style.backgroundColor = "red";
          }
  
  
          DUtest.push(DU);
          UDtest.push(UD);
          DDtest.push(DD);
          DDtest.push(UU);
  
          if (typeof dunn == "undefined") {
            //console.log("No Neural Network create - Failure!!!");
          }
  
          //Calculate and Display Neural Network predictions
          
           
            dupred = dunn.predict([DU]);
            udpred = udnn.predict([UD]);
            ddpred = ddnn.predict([DD]);
            uupred = uunn.predict([UU]);

            //console.log(Math.round(smd_nn(DUseq, DU, dupred[0])),Math.round(smd_nn(UDseq, UD, udpred[0])),Math.round(smd_nn(DDseq, DD, ddpred[0])), Math.round(smd_nn(UUseq, UU, uupred[0])));

            if(Math.round(smd_nn(DUseq, DU, dupred[0])) < 62 && Math.round(smd_nn(UDseq, UD, udpred[0]))< 62 && Math.round(smd_nn(DDseq, DD, ddpred[0])) < 62 && Math.round(smd_nn(UUseq, UU, uupred[0])) < 62){
              document.getElementById("Autenticidad2").innerHTML = "Es auténtico"
              document.getElementById("Autenticidad2").style.visibility = "visible";
              document.getElementById("Autenticidad2").style.backgroundColor = "green";
            }
            else {
              document.getElementById("Autenticidad2").innerHTML = "Es un atacante"
              document.getElementById("Autenticidad2").style.visibility = "visible";
              document.getElementById("Autenticidad2").style.backgroundColor = "red";
            }
 
           
          //Calculate and Display Neuro-Evolutionary Network predictions
       
  
            dunepred = dunenn.activate(DU);
            udnepred = udnenn.activate(UD);
            ddnepred = ddnenn.activate(DD);
            uunepred = uunenn.activate(UU);

            if( Math.round(smd_nn(DUseq, DU, dunepred)) < 39 && Math.round(smd_nn(UDseq, UD, udnepred))  < 39 && Math.round(smd_nn(DDseq, DD, ddnepred)) < 39 && Math.round(smd_nn(UUseq, UU, uunepred)) < 39){
              document.getElementById("Autenticidad3").innerHTML = "Es auténtico"
              document.getElementById("Autenticidad3").style.visibility = "visible";
              document.getElementById("Autenticidad3").style.backgroundColor = "green";
            }
            else {
              document.getElementById("Autenticidad3").innerHTML = "Es un atacante"
              document.getElementById("Autenticidad3").style.visibility = "visible";
              document.getElementById("Autenticidad3").style.backgroundColor = "red";
            }

          
  
          DU = [];
          UD = [];
          DD = [];
          UU = [];
  
          lastdown = lastup = -1;
          return;
        }
  
        if (reEntrenamiento) {
          if (DUseq.length > 24) {
            saveData();
            loadData();
            reEntrenamiento = false;
          }
        }
  
  
  
        try {         
  
          DUseq.push(DU);
          UDseq.push(UD);
          DDseq.push(DD);
          UUseq.push(UU);     
          DU = [];
          UD = [];
          DD = [];
          UU = [];  
          lastdown = lastup = -1;
        }
        catch (e) {
          DU = [];
          UD = [];
          DD = [];
          UU = [];  
          lastdown = lastup = -1;
        }
  
      }
      return;
    }
  
    var temp;
    if (e.type === "keydown") {
      currText += e.key;
      // caso anormal 1 [-,+,-,-]
      if (lastup >= 0 && lastdown > lastup) {
        temp = (Math.random() * (e.timeStamp - lastdown)) + 1
        upgen = lastdown + temp;
        lastup = upgen;
        tecleo.push(upgen);
        flechas.push("+>");
      }
      // caso anormal 2[-,-]
      else if (lastup <= 0 && lastdown > lastup) {
        temp = (Math.random() * (e.timeStamp - lastdown)) + 1
        upgen = lastdown + temp;
        lastup = upgen;
        tecleo.push(upgen);
        flechas.push(">+");
      }
      lastdown = e.timeStamp;
      tecleo.push(e.timeStamp);
      flechas.push("-");

    }
  
    else if (e.type === "keyup") {
      //caso [-,+,+]
      if (lastdown >= 0 && lastdown <= lastup) {
        //console.log("arriba arriba");
        return;
      }
  
      lastup = e.timeStamp;
      tecleo.push(e.timeStamp);
      flechas.push("+");
    };
  }


  function saveData() {
    var user = document.getElementById("usuario").value;
    var jsonDUseq = JSON.stringify(DUseq);
    localStorage.setItem(user + 'DatosDU', jsonDUseq);
  
    var jsonUDseq = JSON.stringify(UDseq);
    localStorage.setItem(user + 'DatosUD', jsonUDseq);
  
    var jsonDDseq = JSON.stringify(DDseq);
    localStorage.setItem(user + 'DatosDD', jsonDDseq);
  
    var jsonUUseq = JSON.stringify(UUseq);
    localStorage.setItem(user + 'DatosUU', jsonUUseq);
  
  
    localStorage.setItem(user + 'DatosPass', testText);
  
    var jsonDUNN = JSON.stringify(dunn);
    localStorage.setItem(user + 'DatosDUNN', jsonDUNN);
  
    var jsonUDNN = JSON.stringify(udnn);
    localStorage.setItem(user + 'DatosUDNN', jsonUDNN);
  
    var jsonDDNN = JSON.stringify(ddnn);
    localStorage.setItem(user + 'DatosDDNN', jsonDDNN);
  
    var jsonUUNN = JSON.stringify(uunn);
    localStorage.setItem(user + 'DatosUUNN', jsonUUNN);
  
    var jsonDUNNNE = JSON.stringify(dunenn);
    localStorage.setItem(user + 'DatosDUNNNE', jsonDUNNNE);
  
    var jsonUDNNNE = JSON.stringify(udnenn);
    localStorage.setItem(user + 'DatosUDNNNE', jsonUDNNNE);
  
    var jsonDDNNNE = JSON.stringify(ddnenn);
    localStorage.setItem(user + 'DatosDDNNNE', jsonDDNNNE);
  
    var jsonUUNNNE = JSON.stringify(uunenn);
    localStorage.setItem(user + 'DatosUUNNNE', jsonUUNNNE);
  
  }
  
  function loadData() {
    //console.log("work");
    user = localStorage.getItem('usuarioLogInCorreo');
    var temp = [];

    var jsonDatos = localStorage.getItem(user + 'DatosPass');
    testText = jsonDatos;
  
    var jsonDuseq = localStorage.getItem(user + 'DatosDU');
    temp = JSON.parse(jsonDuseq);
    DUseq = temp;
  
    var jsonUDseq = localStorage.getItem(user + 'DatosUD');
    temp = JSON.parse(jsonUDseq);
    UDseq = temp;
  
    var jsonDDseq = localStorage.getItem(user + 'DatosDD');
    temp = JSON.parse(jsonDDseq);
    DDseq = temp;
  
    var jsonUUseq = localStorage.getItem(user + 'DatosUU');
    temp = JSON.parse(jsonUUseq);
    UUseq = temp;
  
    var jsonDUNN = localStorage.getItem(user + 'DatosDUNN');
    temp = JSON.parse(jsonDUNN);
    dunn1 = temp;
  
    var jsonUDNN = localStorage.getItem(user + 'DatosUDNN');
    temp = JSON.parse(jsonUDNN);
    udnn1 = temp;
  
    var jsonDDNN = localStorage.getItem(user + 'DatosDDNN');
    temp = JSON.parse(jsonDDNN);
    ddnn1 = temp;
  
    var jsonUUNN = localStorage.getItem(user + 'DatosUUNN');
    temp = JSON.parse(jsonUUNN);
    uunn1 = temp;
  
    var jsonDUNNNE = localStorage.getItem(user + 'DatosDUNNNE');
    temp = JSON.parse(jsonDUNNNE);
    dunnne1 = temp;
  
    var jsonUDNNNE = localStorage.getItem(user + 'DatosUDNNNE');
    temp = JSON.parse(jsonUDNNNE);
    udnnne1 = temp;
  
    var jsonDDNNNE = localStorage.getItem(user + 'DatosDDNNNE');
    temp = JSON.parse(jsonDDNNNE);
    ddnnne1 = temp;
  
    var jsonUUNNNE = localStorage.getItem(user + 'DatosUUNNNE');
    temp = JSON.parse(jsonUUNNNE);
    uunnne1 = temp;
  
    var l = DUseq[0].length;
    if (l > 0) {
      var l1 = Math.ceil(l / 2);
      var options = { hiddenLayers: [l, l, l, l, l], iterations: 1500 };
      dunn = new ML.FNN(dunn1);
      udnn = new ML.FNN(udnn1);
      ddnn = new ML.FNN(uunn1);
      uunn = new ML.FNN(ddnn1);
  
      dunenn = neataptic.Network.fromJSON(dunnne1);
      udnenn = neataptic.Network.fromJSON(udnnne1);
      ddnenn = neataptic.Network.fromJSON(ddnnne1);
      uunenn = neataptic.Network.fromJSON(uunnne1);

    }  
    document.getElementById("testCheck").checked = true;
  
  
  }
  


  function calculardatos(hola) {


    if (hola.length != 0) {
  
      ldown = -1;
      lup = -1;
      for (i = 0; i < hola.length - 1;) {
        tempd = hola[i];
        tempu = hola[i + 1];
        if (ldown >= 0 && lup > 0);
        {
          DU.push(tempu - tempd);
          UD.push(tempd - lup);
          DD.push(tempd - ldown);
          UU.push(tempu - lup);
        }
        if (ldown <= 0 && lup <= 0) {
          DU.push(tempu - tempd);
        }
        lup = tempu;
        ldown = tempd;
        i = i + 2;
  
      }
    }
  }
  



window.onload = function () { 
    user = localStorage.getItem('usuarioLogInCorreo');
  
    function saveData() {
      var user = document.getElementById("usuario").value;
      var jsonDUseq = JSON.stringify(DUseq);
      localStorage.setItem(user + 'DatosDU', jsonDUseq);
  
      var jsonUDseq = JSON.stringify(UDseq);
      localStorage.setItem(user + 'DatosUD', jsonUDseq);
  
      var jsonDDseq = JSON.stringify(DDseq);
      localStorage.setItem(user + 'DatosDD', jsonDDseq);
  
      var jsonUUseq = JSON.stringify(UUseq);
      localStorage.setItem(user + 'DatosUU', jsonUUseq);
  
  
      localStorage.setItem(user + 'DatosPass', testText);
  
      var jsonDUNN = JSON.stringify(dunn);
      localStorage.setItem(user + 'DatosDUNN', jsonDUNN);
  
      var jsonUDNN = JSON.stringify(udnn);
      localStorage.setItem(user + 'DatosUDNN', jsonUDNN);
  
      var jsonDDNN = JSON.stringify(ddnn);
      localStorage.setItem(user + 'DatosDDNN', jsonDDNN);
  
      var jsonUUNN = JSON.stringify(uunn);
      localStorage.setItem(user + 'DatosUUNN', jsonUUNN);
  
      var jsonDUNNNE = JSON.stringify(dunenn);
      localStorage.setItem(user + 'DatosDUNNNE', jsonDUNNNE);
  
      var jsonUDNNNE = JSON.stringify(udnenn);
      localStorage.setItem(user + 'DatosUDNNNE', jsonUDNNNE);
  
      var jsonDDNNNE = JSON.stringify(ddnenn);
      localStorage.setItem(user + 'DatosDDNNNE', jsonDDNNNE);
  
      var jsonUUNNNE = JSON.stringify(uunenn);
      localStorage.setItem(user + 'DatosUUNNNE', jsonUUNNNE);
  
    }
    
  
    function loadData() {
      //console.log("why");
      
      var temp = [];
  
      var jsonDuseq = localStorage.getItem(user + 'DatosDU');
      temp = JSON.parse(jsonDuseq);
      DUseq = temp;
  
  
      var jsonUDseq = localStorage.getItem(user + 'DatosUD');
      temp = JSON.parse(jsonUDseq);
      UDseq = temp;
  
      var jsonDDseq = localStorage.getItem(user + 'DatosDD');
      temp = JSON.parse(jsonDDseq);
      DDseq = temp;
  
      var jsonUUseq = localStorage.getItem(user + 'DatosUU');
      temp = JSON.parse(jsonUUseq);
      UUseq = temp;
  
      var jsonDatos = localStorage.getItem(user + 'DatosPass');
      testText = jsonDatos;
  
  
      var jsonDUNN = localStorage.getItem(user + 'DatosDUNN');
      temp = JSON.parse(jsonDUNN);
      dunn1 = temp;
    
      var jsonUDNN = localStorage.getItem(user + 'DatosUDNN');
      temp = JSON.parse(jsonUDNN);
      udnn1 = temp;
    
      var jsonDDNN = localStorage.getItem(user + 'DatosDDNN');
      temp = JSON.parse(jsonDDNN);
      ddnn1 = temp;
    
      var jsonUUNN = localStorage.getItem(user + 'DatosUUNN');
      temp = JSON.parse(jsonUUNN);
      uunn1 = temp;
    
      var jsonDUNNNE = localStorage.getItem(user + 'DatosDUNNNE');
      temp = JSON.parse(jsonDUNNNE);
      dunnne1 = temp;
    
      var jsonUDNNNE = localStorage.getItem(user + 'DatosUDNNNE');
      temp = JSON.parse(jsonUDNNNE);
      udnnne1 = temp;
    
      var jsonDDNNNE = localStorage.getItem(user + 'DatosDDNNNE');
      temp = JSON.parse(jsonDDNNNE);
      ddnnne1 = temp;
    
      var jsonUUNNNE = localStorage.getItem(user + 'DatosUUNNNE');
      temp = JSON.parse(jsonUUNNNE);
      uunnne1 = temp;
    
      var l = DUseq[0].length;
      if (l > 0) {
        var l1 = Math.ceil(l / 2);
        var options = { hiddenLayers: [l, l, l, l, l], iterations: 1500 };
        dunn = new ML.FNN(dunn1);
        udnn = new ML.FNN(udnn1);
        ddnn = new ML.FNN(uunn1);
        uunn = new ML.FNN(ddnn1);
  
        dunenn = neataptic.Network.fromJSON(dunnne1);
        udnenn = neataptic.Network.fromJSON(udnnne1);
        ddnenn = neataptic.Network.fromJSON(ddnnne1);
        uunenn = neataptic.Network.fromJSON(uunnne1);
  
      }
   
      document.getElementById("testCheck").checked = true;
      document.getElementById("contenedor").onclick = loadData();
     
  
  
    } 
  
  
 
  
  };
  document.onkeydown = captureKeyEvent;
  document.onkeyup = captureKeyEvent;  
  document.onkeypress = function (e) {
    e = e || window.event;
    var charCode = e.keyCode || e.which;
    if (charCode === 32) {
      e.preventDefault();
      return false;
    }
  }
