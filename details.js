function displayTacticInfo (hyps, goal, predicted) {
    overlay = document.getElementById("overlay")
    overlay.style.display = "block";
    overlay.style.top = window.pageYOffset;
    overlay.style.height = window.innerHeight * 0.9
    predictedDiv = document.getElementById("predicted")
    predictedDiv.innerHTML = "<h3>Predicted</h3> <pre id='tactic'>" + predicted + "</pre>"
    contextDiv = document.getElementById("context")
    contextDiv.innerHTML = "<h3>Context:</h3>" +
        "<pre id='hyps'>" + hyps + "</pre>" +
        "<hr>" +
        "<pre id='goal'>" + goal + "</pre>";
}
function hideTacticInfo () {
    document.getElementById("overlay").style.display = "none";
}
