/*!
 * Keystroke Dynamics
 *
 * This file is licensed under the MIT license
 *
 *   Vikas Desai - vikasdesai.github.io
 *
 */


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
var user= "";

function createDefaultChart(ctx, title, maxy) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      datasets: []
    },
    options: {
      title: {
        display: true,
        text: title
      },
      legend: {
        display: false
      },
      scales: {
        yAxes: [{
          ticks: {
            //beginAtZero:true
            min: 0,
            max: maxy
          }
        }]
      }
    }
  });
}

function createScatterChart(ctx, title) {
  return new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: []
    },
    options: {
      title: {
        display: true,
        text: title
      },
      legend: {
        display: false
      },
      scales: {
        xAxes: [{
          ticks: {
            beginAtZero: true
          }
        }],
        yAxes: [{
          ticks: {
            beginAtZero: true
          }
        }]
      }
    }
  });
}

async function entrenarNeuronas() {

    var l = DUseq[0].length;
    if (l > 0) {
      var l1 = Math.ceil(l / 2);
      var l2 = Math.ceil(l1 / 2);
      var options = { hiddenLayers: [l, l, l, l, l], iterations: 1500 };
      //var options = { hiddenLayers: [3] };

      dunn = new ML.FNN(options);
      dunn.activation = "exponential-elu"; //rectified linear unit (ReLU)
      dunn.train(DUseq, DUseq);

      udnn = new ML.FNN(options);
      udnn.activation = "exponential-elu";
      udnn.train(UDseq, UDseq);

      ddnn = new ML.FNN(options);
      ddnn.activation = "exponential-elu";
      ddnn.train(DDseq, DDseq);

      uunn = new ML.FNN(options);
      uunn.activation = "exponential-elu";
      uunn.train(UUseq, UUseq);

      //Neuro-Evolutionary Network - neataptic.js
      dunedata = DUseq.map(function (data) { return { input: data, output: data }; });
      udnedata = UDseq.map(function (data) { return { input: data, output: data }; });
      ddnedata = DDseq.map(function (data) { return { input: data, output: data }; });
      uunedata = UUseq.map(function (data) { return { input: data, output: data }; });

      dunenn = neataptic.architect.Perceptron(l, l1, l);
      udnenn = neataptic.architect.Perceptron(l - 1, l1, l - 1);
      ddnenn = neataptic.architect.Perceptron(l - 1, l1, l - 1);
      uunenn = neataptic.architect.Perceptron(l - 1, l1, l - 1);

      neopts = {
        mutation: neataptic.methods.mutation.FFW,
        equal: true,
        popsize: 100,
        elitism: 10,
        log: 500,
        error: 150, //0.03,
        iterations: 1500,
        mutationRate: 0.4
      };

      neDone = await Promise.all([
        dunenn.evolve(dunedata, neopts),
        udnenn.evolve(udnedata, neopts),
        ddnenn.evolve(ddnedata, neopts),
        uunenn.evolve(uunedata, neopts)
      ]);
      console.log(neDone);
      saveData();
    
  }
}

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

function removeLastFromChart(myChart) {
  myChart.data.datasets.pop();
  myChart.update();
}

function addDataToChart(myChart, newdata, color, showline) {
  myChart.data.datasets.push({
    label: "",
    data: newdata,
    backgroundColor: color,
    borderColor: color,
    fill: false,
    pointRadius: 3,
    pointHoverRadius: 9,
    showLine: showline
  });
  myChart.update();
}




function captureKeyEvent(e) {
 

  if (e.key.length > 1) {
    if (e.key === "Backspace") {
      txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
      txtctx.strokeText(testText, 0, txtCanvas.height / 2);
      currText = "";
      flechas = [];
      tecleo = [];
      DU = [];
      UD = [];
      DD = [];
      UU = [];
      return
    }
    //se detecto enter ( final de clave )
    if (e.key === "Enter" && e.type === "keyup") {
      calculardatos(tecleo);
      UD.shift();
      DD.shift();
      DU.shift();
      UU.shift();    
      flechas = [];
      tecleo = [];


      //problemas
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
            console.log("tipo 1")
            console.log(UD, DU, DD, UU);
            txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
            txtctx.strokeText(testText, 0, txtCanvas.height / 2);
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


        else if (DD.length <= DU.length - 2 || UD.length <= DU.length - 2) {
          alert("Se produjo un error.");
          console.log("tipo 2")
          console.log(UD, DU, DD, UU);
          txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
          txtctx.strokeText(testText, 0, txtCanvas.height / 2);
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
          console.log("tipo 3")
          console.log(UD, DU, DD, UU);
          txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
          txtctx.strokeText(testText, 0, txtCanvas.height / 2);
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
        txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
        txtctx.strokeText(testText, 0, txtCanvas.height / 2);
        currText = "";
        lastdown = lastup = -1;
        return;
      }

      txtctx.clearRect(0, 0, txtCanvas.width, txtCanvas.height);
      txtctx.strokeText(testText, 0, txtCanvas.height / 2);

      currText = "";



      duavg = math.mean(DU);
      udavg = math.mean(UD);
      ddavg = math.mean(DD);
      uuavg = math.mean(UU);            
        
 
    

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
    document.getElementById("count").innerHTML = 21 - DUseq.length;
  if(DUseq.length > 20)
  {
    document.getElementById("saveData").hidden =false;


    alert("Se ha entrenado, se a activado la opcion de guardar contraseña. esto termina el registro")    
  }
    return;
  }

  var temp;
  if (e.type === "keydown") {
    currText += e.key;        
    // caso anormal 1 [-,+,-,-]
   if (lastup >= 0 && lastdown > lastup) {
    temp = (Math.random() * 20)
    upgen = e.timeStamp - temp
    lastup = upgen;
    tecleo.push(upgen);
    flechas.push("+>");
    }
    // caso anormal 2[-,-]
    else if (lastup <= 0 && lastdown > lastup) {
      temp = (Math.random() * 20)
      upgen = e.timeStamp - temp
      lastup = upgen;
      tecleo.push(upgen);
      flechas.push(">+");    
    }
    lastdown = e.timeStamp;
    tecleo.push(e.timeStamp);
    flechas.push("-");
    txtctx.fillText(currText, 0, txtCanvas.height / 2);
  }

  else if (e.type === "keyup") {
    //caso [-,+,+]
    if (lastdown >= 0 && lastdown <= lastup) {
      console.log("arriba arriba");
      return;
    }
  
    lastup = e.timeStamp;
    tecleo.push(e.timeStamp);
    flechas.push("+");
  }
  
}



function reEntrenar() { 
  reEntrenamiento = true;
  while (DUseq.length > 15) {
    DUseq.shift();
    UDseq.shift();
    DDseq.shift();
    UUseq.shift();

  }
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

  var jsonDUNNNE = JSON.stringify(dunnne);
  localStorage.setItem(user + 'DatosDUNNNE', jsonDUNNNE);

  var jsonUDNNNE = JSON.stringify(udnnne);
  localStorage.setItem(user + 'DatosUDNNNE', jsonUDNNNE);

  var jsonDDNNNE = JSON.stringify(ddnnne);
  localStorage.setItem(user + 'DatosDDNNNE', jsonDDNNNE);

  var jsonUUNNNE = JSON.stringify(uunnne);
  localStorage.setItem(user + 'DatosUUNNNE', jsonUUNNNE);

  document.getElementById("continuar").hidden =false;
}

function loadData() {
 

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

 

}

window.onload = function () {
  
  txtCanvas = document.getElementById("textEntry");
  txtctx = txtCanvas.getContext("2d");
  txtctx.font = "100px verdana"; 

  user = localStorage.getItem('usuarioNuevoCorreo');
  

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

  document.getElementById("saveData").onclick = entrenarNeuronas;

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

 