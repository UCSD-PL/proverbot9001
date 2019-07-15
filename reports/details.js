/*##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################*/
selectedIdx = -1
function initRegions(){
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
}
function init() {
    setSelectedIdx()
    initRegions()
}
function setSelectedIdx(){
    if (window.location.hash != ""){
        idx = /\#command-(\d+-?\d*)/.exec(window.location.hash)[1]
        collapsible_idx_match = /(\d+)-?\d*/.exec(idx)
        if (collapsible_idx_match != null){
            regionIdx = collapsible_idx_match[1]
            var button = document.getElementById("collapsible-"+regionIdx);
            button.classList.add("active")
            content = button.nextElementSibling
            content.style.maxHeight = content.scrollHeight + "px";
        }
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
    items = string.split("% ")
    if (items.length == 1 && items[0] == "")
        return []
    else
        return items
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
    if (predictions.length > 0){
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
    } else {
        statsDiv.innerHTML = ""
    }

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
    if (idx == selectedIdx){
        deselectTactic()
    } else {
        if (selectedIdx != 1){
            deselectTactic()
        }
        selectedIdx = idx
        displayTacticInfo(idx)
        tacSpan = document.getElementById("command-" + idx)
        tacSpan.style.backgroundColor = "LightCyan"
    }
}
function deselectTactic() {
    tacSpan = document.getElementById("command-" + selectedIdx)
    if (tacSpan != null){
        tacSpan.style.backgroundColor = ""
        hideTacticInfo()
    }
    selectedIdx = -1;
}
