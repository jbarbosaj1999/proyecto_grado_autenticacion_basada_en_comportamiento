


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
            beginAtZero: true

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

function cargarNeuronas() {

}

async function entrenarNeuronas() {


  if (document.getElementById("testCheck").checked) {
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
    }
  }
}

function initCharts() {
  ductx = document.getElementById("DUChart").getContext('2d');
  DUChart = createDefaultChart(ductx, "DOWN-UP (Dwell Time)", 200);

  udctx = document.getElementById("UDChart").getContext('2d');
  UDChart = createDefaultChart(udctx, "UP-DOWN (Flight Time)", 200);

  ddctx = document.getElementById("DDChart").getContext('2d');
  DDChart = createDefaultChart(ddctx, "DOWN-DOWN (Speed)", 200);

  uuctx = document.getElementById("UUChart").getContext('2d');
  UUChart = createDefaultChart(uuctx, "UP - UP (Speed)", 200);

  txtCanvas = document.getElementById("textEntry");
  txtctx = txtCanvas.getContext("2d");
  txtctx.font = "100px verdana";

  dunnctx = document.getElementById("DUNNChart").getContext('2d');
  DUNNChart = createDefaultChart(dunnctx, "DOWN-UP (Dwell Time)", 200);

  udnnctx = document.getElementById("UDNNChart").getContext('2d');
  UDNNChart = createDefaultChart(udnnctx, "UP-DOWN (Flight Time)", 200);

  ddnnctx = document.getElementById("DDNNChart").getContext('2d');
  DDNNChart = createDefaultChart(ddnnctx, "DOWN-DOWN (Speed)", 200);

  uunnctx = document.getElementById("UUNNChart").getContext('2d');
  UUNNChart = createDefaultChart(uunnctx, "UP-UP (Speed)", 200);

  dunectx = document.getElementById("DUNEChart").getContext('2d');
  DUNEChart = createDefaultChart(dunectx, "DOWN-UP (Dwell Time)", 200);

  udnectx = document.getElementById("UDNEChart").getContext('2d');
  UDNEChart = createDefaultChart(udnectx, "UP-DOWN (Flight Time)", 200);

  ddnectx = document.getElementById("DDNEChart").getContext('2d');
  DDNEChart = createDefaultChart(ddnectx, "DOWN-DOWN (Speed)", 200);

  uunectx = document.getElementById("UUNEChart").getContext('2d');
  UUNEChart = createDefaultChart(uunectx, "UP-UP (Speed)", 200);

  duudctx = document.getElementById("DU-UD").getContext('2d');
  DUUDChart = createScatterChart(duudctx, "DU-UD");

  udddctx = document.getElementById("UD-DD").getContext('2d');
  UDDDChart = createScatterChart(udddctx, "UD-DD");

  ddductx = document.getElementById("DD-DU").getContext('2d');
  DDDUChart = createScatterChart(ddductx, "DD-DU");
}

function showNN() {
  document.getElementById("nnheader").style.visibility = "visible";
  document.getElementById("nntable").style.visibility = "visible";

  if (document.getElementById("testCheck").checked == false) {
    document.getElementById("nnpredict").checked = false;
    alert("Please enable Test for this to work");
  }
}


function showNE() {
  if (document.getElementById("testCheck").checked == false) {
    document.getElementById("nepredict").checked = false;
    alert("Please enable Test for this to work");
    return;
  }
  // if (typeof neDone == "undefined") {
  //   alert("Sorry Still Evolving!!!");
  //   document.getElementById("nepredict").checked = false;
  // } else {

  // }
  document.getElementById("neheader").style.visibility = "visible";
  document.getElementById("netable").style.visibility = "visible";
}

function updateMinMaxTrain() {
  var maxTrainError = maxsmd(DUseq, UDseq, DDseq, UUseq);
  var minTrainError = minsmd(DUseq, UDseq, DDseq, UUseq);

  document.getElementById("dutrain").innerHTML = (minTrainError[0]) + "-" + (maxTrainError[0]);
  document.getElementById("udtrain").innerHTML = (minTrainError[1]) + "-" + (maxTrainError[1]);
  document.getElementById("ddtrain").innerHTML = (minTrainError[2]) + "-" + (maxTrainError[2]);
}

function captureKeyEvent(e) {

  //newevent = {event :  e.type, code : e.code, keycode :  e.keyCode, keytime : e.timeStamp };
  //keyevents.push(newevent);

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
    if (e.key === "Enter" && e.type === "keyup") {
      calculardatos(tecleo);
      UD.shift();
      DD.shift();
      DU.shift();
      UU.shift();
      console.log(DU, UD, DD, UU);
      console.log(flechas);
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

      var test = document.getElementById("testCheck").checked;

      if (test) {
        document.getElementById("dutest").innerHTML = (scaledManhattanDist(DUseq, DU));
        document.getElementById("udtest").innerHTML = (scaledManhattanDist(UDseq, UD));
        document.getElementById("ddtest").innerHTML = (scaledManhattanDist(DDseq, DD));
        document.getElementById("uutest").innerHTML = (scaledManhattanDist(UUseq, UU));

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


        document.getElementById("dutrain").innerHTML = trainValueDU;
        document.getElementById("udtrain").innerHTML = trainValueUD;
        document.getElementById("ddtrain").innerHTML = trainValueDD;
        document.getElementById("uutrain").innerHTML = trainValueUU;

        var stdDU = (math.std(DUseq.map(smd_for_path)) * 2);

        var stdUD = (math.std(UDseq.map(smd_for_path)) * 2);

        var stdDD = (math.std(DDseq.map(smd_for_path)) * 2);

        var stdUU = (math.std(UUseq.map(smd_for_path)) * 2);

        console.log(Math.abs((scaledManhattanDist(DUseq, DU))) + " - " + trainValueDU + " :: " + stdDU);
        console.log(Math.abs((scaledManhattanDist(UDseq, UD))) + " - " + trainValueUD + " :: " + stdUD);
        console.log(Math.abs((scaledManhattanDist(DDseq, DD))) + " - " + trainValueDD + " :: " + stdDD);
        console.log(Math.abs((scaledManhattanDist(UUseq, UU))) + " - " + trainValueUU + " :: " + stdUU);

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

          document.getElementById("count").innerHTML = DUseq.length;
          if (typeof dunn == "undefined") {
            console.log("No Neural Network create - Failure!!!");
          }
        }
        else {
          document.getElementById("Autenticidad").innerHTML = "Es un atacante"
          document.getElementById("Autenticidad").style.visibility = "visible";
          document.getElementById("Autenticidad").style.backgroundColor = "red";
        }


        if (DUChart.data.datasets.length > DUseq.length + 1) {
          removeLastFromChart(DUChart);
          removeLastFromChart(UDChart);
          removeLastFromChart(DDChart);
          removeLastFromChart(UUChart);

          removeLastFromChart(DUUDChart);
          removeLastFromChart(UDDDChart);
          removeLastFromChart(DDDUChart);
        }
        addDataToChart(DUChart, DU, 'rgba(0,0,255,0.5)', true);
        addDataToChart(UDChart, UD, 'rgba(0,0,255,0.5)', true);
        addDataToChart(DDChart, DD, 'rgba(0,0,255,0.5)', true);
        addDataToChart(UUChart, UU, 'rgba(0,0,255,0.5)', true);

        addDataToChart(DUUDChart, [{ x: Math.round(duavg), y: Math.round(udavg) }], 'rgba(0,0,255,1)', false);
        addDataToChart(UDDDChart, [{ x: Math.round(udavg), y: Math.round(ddavg) }], 'rgba(0,0,255,1)', false);
        addDataToChart(DDDUChart, [{ x: Math.round(ddavg), y: Math.round(duavg) }], 'rgba(0,0,255,1)', false);

        DUtest.push(DU);
        UDtest.push(UD);
        DDtest.push(DD);
        DDtest.push(UU);

        if (typeof dunn == "undefined") {
          console.log("No Neural Network create - Failure!!!");
        }

        //Calculate and Display Neural Network predictions
        if (document.getElementById("nnpredict").checked) {
          DUNNChart.data.datasets = [];
          UDNNChart.data.datasets = [];
          DDNNChart.data.datasets = [];
          UUNNChart.data.datasets = [];

          dupred = dunn.predict([DU]);
          udpred = udnn.predict([UD]);
          ddpred = ddnn.predict([DD]);
          uupred = uunn.predict([UU]);

          document.getElementById("dunnpred").innerHTML = Math.round(smd_nn(DUseq, DU, dupred[0]));
          document.getElementById("udnnpred").innerHTML = Math.round(smd_nn(UDseq, UD, udpred[0]));
          document.getElementById("ddnnpred").innerHTML = Math.round(smd_nn(DDseq, DD, ddpred[0]));
          document.getElementById("uunnpred").innerHTML = Math.round(smd_nn(UUseq, UU, uupred[0]));

          document.getElementById("dunnsmd").innerHTML = document.getElementById("dutest").innerHTML;
          document.getElementById("udnnsmd").innerHTML = document.getElementById("udtest").innerHTML;
          document.getElementById("ddnnsmd").innerHTML = document.getElementById("ddtest").innerHTML;
          document.getElementById("uunnsmd").innerHTML = document.getElementById("uutest").innerHTML;

          addDataToChart(DUNNChart, dupred[0], 'rgba(0,255,0,0.5)', true);
          addDataToChart(UDNNChart, udpred[0], 'rgba(0,255,0,0.5)', true);
          addDataToChart(DDNNChart, ddpred[0], 'rgba(0,255,0,0.5)', true);
          addDataToChart(UUNNChart, uupred[0], 'rgba(0,255,0,0.5)', true);

          addDataToChart(DUNNChart, DU, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UDNNChart, UD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(DDNNChart, DD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UUNNChart, UU, 'rgba(0,0,255,0.5)', true);

          //addDataToChart(DUNNChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(UDNNChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(DDNNChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
        }

        //Calculate and Display Neuro-Evolutionary Network predictions
        if (document.getElementById("nepredict").checked) {
          DUNEChart.data.datasets = [];
          UDNEChart.data.datasets = [];
          DDNEChart.data.datasets = [];
          UUNEChart.data.datasets = [];

          dunepred = dunenn.activate(DU);
          udnepred = udnenn.activate(UD);
          ddnepred = ddnenn.activate(DD);
          uunepred = uunenn.activate(UU);

          document.getElementById("dunepred").innerHTML = Math.round(smd_nn(DUseq, DU, dunepred));
          document.getElementById("udnepred").innerHTML = Math.round(smd_nn(UDseq, UD, udnepred));
          document.getElementById("ddnepred").innerHTML = Math.round(smd_nn(DDseq, DD, ddnepred));
          document.getElementById("uunepred").innerHTML = Math.round(smd_nn(UUseq, UU, ddnepred));

          document.getElementById("dunnpred_").innerHTML = document.getElementById("dunnpred").innerHTML;
          document.getElementById("udnnpred_").innerHTML = document.getElementById("udnnpred").innerHTML;
          document.getElementById("ddnnpred_").innerHTML = document.getElementById("ddnnpred").innerHTML;
          document.getElementById("uunnpred_").innerHTML = document.getElementById("uunnpred").innerHTML;

          document.getElementById("dunnsmd_").innerHTML = document.getElementById("dutest").innerHTML;
          document.getElementById("udnnsmd_").innerHTML = document.getElementById("udtest").innerHTML;
          document.getElementById("ddnnsmd_").innerHTML = document.getElementById("ddtest").innerHTML;
          document.getElementById("uunnsmd_").innerHTML = document.getElementById("uutest").innerHTML;

          addDataToChart(DUNEChart, dunepred, 'rgba(0,0,0,0.5)', true);
          addDataToChart(UDNEChart, udnepred, 'rgba(0,0,0,0.5)', true);
          addDataToChart(DDNEChart, ddnepred, 'rgba(0,0,0,0.5)', true);
          addDataToChart(UUNEChart, uunepred, 'rgba(0,0,0,0.5)', true);

          addDataToChart(DUNEChart, DU, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UDNEChart, UD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(DDNEChart, DD, 'rgba(0,0,255,0.5)', true);
          addDataToChart(UUNEChart, UU, 'rgba(0,0,255,0.5)', true);

          //addDataToChart(DUNEChart, dupred[0], 'rgba(0,255,0,0.5)', true);
          //addDataToChart(UDNEChart, udpred[0], 'rgba(0,255,0,0.5)', true);
          //addDataToChart(DDNEChart, ddpred[0], 'rgba(0,255,0,0.5)', true);

          //addDataToChart(DUNEChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(UDNEChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
          //addDataToChart(DDNEChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
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
        document.getElementById("dutrain").innerHTML = (scaledManhattanDist(DUseq, DU));
        document.getElementById("udtrain").innerHTML = (scaledManhattanDist(UDseq, UD));
        document.getElementById("ddtrain").innerHTML = (scaledManhattanDist(DDseq, DD));
        document.getElementById("uutrain").innerHTML = (scaledManhattanDist(UUseq, UU));
        document.getElementById("count").innerHTML = DUseq.length;

        DUseq.push(DU);
        UDseq.push(UD);
        DDseq.push(DD);
        UUseq.push(UU);

        if (DUChart.data.datasets.length > 1) {
          removeLastFromChart(DUChart);
          removeLastFromChart(UDChart);
          removeLastFromChart(DDChart);
        }

        addDataToChart(DUChart, DU, 'rgba(255,0,0,1)', false);
        addDataToChart(UDChart, UD, 'rgba(255,0,0,1)', false);
        addDataToChart(DDChart, DD, 'rgba(255,0,0,1)', false);

        addDataToChart(DUChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
        addDataToChart(UDChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
        addDataToChart(DDChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);

        addDataToChart(DUUDChart, [{ x: duavg, y: udavg }], 'rgba(255,0,0,1)', false);
        addDataToChart(UDDDChart, [{ x: udavg, y: ddavg }], 'rgba(255,0,0,1)', false);
        addDataToChart(DDDUChart, [{ x: ddavg, y: duavg }], 'rgba(255,0,0,1)', false);

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
  };
}

function reEntrenar() {
  document.getElementById("testCheck").checked = false;
  reEntrenamiento = true;
  while (DUseq.length > 15) {
    DUseq.shift();
    UDseq.shift();
    DDseq.shift();
    UUseq.shift();

  }
}

function reEntrenarTot() {
  document.getElementById("testCheck").checked = false;
  reEntrenamiento = true;
  while (DUseq.length > 0) {
    DUseq.shift();
    UDseq.shift();
    DDseq.shift();
    UUseq.shift();

  }
}

function habilitar() {
  if (document.getElementById("usuario").innerText = ! "") {
    document.getElementById("saveData").disabled = false;
    document.getElementById("loadData").disabled = false;
    document.getElementById("reTrain").disabled = false;
    document.getElementById("reTrainTot").disabled = false;
    return;
  }
  else if (document.getElementById("usuario").innerText = "") {
    document.getElementById("saveData").disabled = true;
    document.getElementById("loadData").disabled = true;
    document.getElementById("reTrain").disabled = true;
    document.getElementById("reTrainTot").disabled = false;
    return;
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
  var user = document.getElementById("usuario").value;
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

  var jsonDUNN = localStorage.getItem(user + 'DatosDUNN');
  temp = JSON.parse(jsonDUNN);
  dunn1 = temp;

  var jsonUDNN = localStorage.getItem(user + 'DatosUDNN');
  temp = JSON.parse(jsonUDNN);
  udnn1 = temp;

  var jsonDDNN = localStorage.getItem(user + 'DatosDDNN');
  temp = JSON.parse(jsonDDNN);
  dunn1 = temp;

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




  addDataToChart(DUChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
  addDataToChart(UDChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
  addDataToChart(DDChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
  addDataToChart(UUChart, meanPath(UUseq), 'rgba(255,0,0,0.5)', true);

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
  console.log(tvDD1, tvDD2)
  var trainValueDD = (tvDD2 + tvDD1) / 2;




  var tvUU1 = minTrainError[3];
  var tvUU2 = maxTrainError[3];
  var trainValueUU = (tvUU2 + tvUU1) / 2;

  document.getElementById("dutrain").innerHTML = (trainValueDU);
  document.getElementById("udtrain").innerHTML = (trainValueUD);
  document.getElementById("ddtrain").innerHTML = (trainValueDD);
  document.getElementById("uutrain").innerHTML = (trainValueUU);

  document.getElementById("testCheck").checked = true;


}


function exportar() {

  EXCEL_TYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8';
  EXCEL_EXTENSION = '.xlsx';

  user = document.getElementById("usuario").value;
  jsonDuseq = localStorage.getItem('joseBDatosDU');
  worksheet = XLSX.utils.json_to_sheet(jsonDuseq);
  workbook = {
    Sheets: {
      'data': worksheet
    }, SheetNames: ['data']
  };
  excelbuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
}


window.onload = function () {
  initCharts();

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

    var user = document.getElementById("usuario").value;
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
  

    addDataToChart(DUChart, meanPath(DUseq), 'rgba(255,0,0,0.5)', true);
    addDataToChart(UDChart, meanPath(UDseq), 'rgba(255,0,0,0.5)', true);
    addDataToChart(DDChart, meanPath(DDseq), 'rgba(255,0,0,0.5)', true);
    addDataToChart(UUChart, meanPath(UUseq), 'rgba(255,0,0,0.5)', true);



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
    console.log(tvDD1, tvDD2)
    var trainValueDD = (tvDD2 + tvDD1) / 2;




    var tvUU1 = minTrainError[3];
    var tvUU2 = maxTrainError[3];
    var trainValueUU = (tvUU2 + tvUU1) / 2;

    document.getElementById("dutrain").innerHTML = (trainValueDU);
    document.getElementById("udtrain").innerHTML = (trainValueUD);
    document.getElementById("ddtrain").innerHTML = (trainValueDD);
    document.getElementById("uutrain").innerHTML = (trainValueUU);


    document.getElementById("count").innerHTML = DUseq.length;
    document.getElementById("testCheck").checked = true;


  }

  document.getElementById("saveData").onclick = saveData;
  document.getElementById("loadData").onclick = loadData;
  document.getElementById("reTrain").onclick = reEntrenar;
  document.getElementById("reTrainTot").onclick = reEntrenarTot;
  document.getElementById("exportar").onclick = exportar;


  document.getElementById("usuario").addEventListener("keyup", habilitar);

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

