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
    return command.trim().split(" ")[0].replace(/\.$/, "")
}

function from_list_string(string){
    return string.split("% ")
}
function displayTacticInfo (idx) {
    overlay = document.getElementById("overlay")
    overlay.style.display = "block";

    tacSpan = document.getElementById("command-" + idx)

    predictions = from_list_string(tacSpan.dataset.predictions)
    certainties = from_list_string(tacSpan.dataset.certainties)
    klasses = from_list_string(tacSpan.dataset.grades)
    predictedDiv = document.getElementById("predicted")
    linkLoc = window.location.protocol + "//" + window.location.hostname
        + window.location.pathname + "#command-" + idx
    predictedDiv.innerHTML =
        "<h3>Predicted <a onclick='window.location.hash = \"\"' href=" + linkLoc + ">[link]</a></h3>"
    for (i = 0; i < predictions.length; ++i){
        var predictionPre = document.createElement("pre")
        predictionPre.appendChild(document.createTextNode(predictions[i]))
        predictionPre.classList.add('tactic')
        predictionPre.classList.add(klasses[i])
        predictedDiv.appendChild(predictionPre)
        var predictedCertaintyP = document.createElement("p")
        predictedCertaintyP.appendChild(document.createTextNode("(" + (certainties[i] * 100).toFixed(2) + "%)"))
        predictedCertaintyP.classList.add('certainty')
        predictedCertaintyP.classList.add(klasses[i])
        predictedDiv.appendChild(predictedCertaintyP)
        var br = document.createElement("br")
        predictedDiv.appendChild(br)
    }

    hyps = tacSpan.dataset.hyps
    goal = tacSpan.dataset.goal
    contextDiv = document.getElementById("context")
    contextDiv.innerHTML = "<h3>Context:</h3>" +
        "<pre id='hyps'>" + hyps + "</pre>" +
        "<hr>" +
        "<pre id='goal'>" + goal + "</pre>";

    num_total = tacSpan.dataset.numTotal
    num_predicteds = from_list_string(tacSpan.dataset.numPredicteds)
    num_corrects = from_list_string(tacSpan.dataset.numCorrects)
    num_actual_corrects = tacSpan.dataset.numActualCorrects
    num_actual_in_file = tacSpan.dataset.numActualInFile
    actual_tactic = tacSpan.dataset.actualTactic
    search_index = tacSpan.dataset.searchIdx
    statsDiv = document.getElementById("stats")
    statsDiv.innerHTML = "Predicted \"<tt>" + getStem(predictions[search_index]) +
        " *</tt>\" " + Math.floor((num_predicteds[search_index] / num_total) * 100) +
        "% of the time (" + num_predicteds[search_index] + "/" + num_total + ")<br>\n" +
        Math.floor((num_corrects[search_index] / num_predicteds[search_index]) * 100) +
        "% of \"<tt>" + getStem(predictions[search_index]) +
        " *</tt>\" predictions are correct (" +
        num_corrects[search_index] + "/" + num_predicteds[search_index] + "). " +
        Math.floor((num_actual_corrects / num_actual_in_file) * 100) +
        "% of \"<tt>" + getStem(actual_tactic) +
        " *</tt>\"'s in file correctly predicted (" +
        num_actual_corrects + "/" + num_actual_in_file + ").";
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
