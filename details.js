selectedIdx = -1
function setSelectedIdx(){
    if (window.location.hash != ""){
        idx = /\#command-(\d+)/.exec(window.location.hash)[1]
        selectTactic(idx)
    }
    window.onhashchange = setSelectedIdx
}
function hoverTactic(idx){
    if (selectedIdx != -1){
        return;
    } else {
        displayTacticInfo(idx)
    }
}
function getStem(command) {
    return command.trip().split(" ")[0].replace(/\.$/, "")
}
function displayTacticInfo (idx) {
    overlay = document.getElementById("overlay")
    overlay.style.display = "block";

    tacSpan = document.getElementById("command-" + idx)

    predicted = tacSpan.dataset.predicted
    predictedDiv = document.getElementById("predicted")
    linkLoc = window.location.protocol + "//" + window.location.hostname
        + window.location.pathname + "#command-" + idx
    predictedDiv.innerHTML = "<h3>Predicted <a onclick='window.location.hash = \"\"' href=" + linkLoc + ">[link]</a></h3> <pre id='tactic'>" + predicted + "</pre>"

    hyps = tacSpan.dataset.hyps
    goal = tacSpan.dataset.goal
    contextDiv = document.getElementById("context")
    contextDiv.innerHTML = "<h3>Context:</h3>" +
        "<pre id='hyps'>" + hyps + "</pre>" +
        "<hr>" +
        "<pre id='goal'>" + goal + "</pre>";

    num_total = tacSpan.dataset.numTotal
    num_predicted = tacSpan.dataset.numPredicted
    num_correct = tacSpan.dataset.numCorrect
    statsDiv = document.getElementById("stats")
    statsDiv.innerHTML = "Predicted \"<tt>" + getStem(predicted) + "</tt>\" " + Math.floor((num_predicted / num_total) * 100) +
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
function selectTactic(idx) {
    if (selectedIdx != 1){
        deselectTactic()
    }
    selectedIdx = idx
    displayTacticInfo(idx)
    tacSpan = document.getElementById("command-" + idx)
    tacSpan.style.backgroundColor = "LightCyan"
}
function deselectTactic() {
    tacSpan = document.getElementById("command-" + selectedIdx)
    if (tacSpan != null){
        tacSpan.style.backgroundColor = ""
        hideTacticInfo()
    }
    selectedIdx = -1;
}
