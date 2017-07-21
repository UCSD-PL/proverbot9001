selectedIdx = -1
function hoverTactic(hyps, goal, predicted){
    if (selectedIdx != -1){
        return;
    } else {
        displayTacticInfo(hyps, goal, predicted)
    }
}
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
function unhoverTactic() {
    if (selectedIdx != -1){
        return;
    }
    hideTacticInfo()
}
function hideTacticInfo () {
    document.getElementById("overlay").style.display = "none";
}
function selectTactic(idx, hyps, goal, predicted) {
    if (selectedIdx != 1){
        deselectTactic()
    }
    selectedIdx = idx
    displayTacticInfo(hyps, goal, predicted)
    tacSpan = document.getElementById("context-" + idx)
    tacSpan.style.backgroundColor = "LightCyan"
}
function deselectTactic() {
    tacSpan = document.getElementById("context-" + selectedIdx)
    tacSpan.style.backgroundColor = ""
    selectedIdx = -1;
    hideTacticInfo()
}
