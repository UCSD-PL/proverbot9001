selectedIdx = -1
function hoverTactic(hyps, goal, predicted, num_predicted, num_correct, num_total){
    if (selectedIdx != -1){
        return;
    } else {
        displayTacticInfo(hyps, goal, predicted, num_predicted, num_correct, num_total)
    }
}
function displayTacticInfo (hyps, goal, predicted, num_predicted, num_correct, num_total) {
    overlay = document.getElementById("overlay")
    overlay.style.display = "block";
    predictedDiv = document.getElementById("predicted")
    predictedDiv.innerHTML = "<h3>Predicted</h3> <pre id='tactic'>" + predicted + "</pre>"
    contextDiv = document.getElementById("context")
    contextDiv.innerHTML = "<h3>Context:</h3>" +
        "<pre id='hyps'>" + hyps + "</pre>" +
        "<hr>" +
        "<pre id='goal'>" + goal + "</pre>";
    statsDiv = document.getElementById("stats")
    statsDiv.innerHTML = "Predicted \"<tt>" + predicted + "</tt>\" " + Math.floor((num_total / num_predicted) * 100) +
        "% of the time (" + num_predicted + "/" + num_total + ")<br>\n" +
        "Correct " + Math.floor((num_correct / num_predicted) * 100) + "% of the time";
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
function selectTactic(idx, hyps, goal, predicted, num_predicted, num_correct, num_total) {
    if (selectedIdx != 1){
        deselectTactic()
    }
    selectedIdx = idx
    displayTacticInfo(hyps, goal, predicted, num_predicted, num_correct, num_total)
    tacSpan = document.getElementById("context-" + idx)
    tacSpan.style.backgroundColor = "LightCyan"
}
function deselectTactic() {
    tacSpan = document.getElementById("context-" + selectedIdx)
    if (tacSpan != null){
        tacSpan.style.backgroundColor = ""
        hideTacticInfo()
    }
    selectedIdx = -1;
}
