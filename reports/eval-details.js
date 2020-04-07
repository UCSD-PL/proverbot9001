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
function addListeners(){
    document.body.addEventListener("click", deselectTactic)

    var tac_spans = document.getElementsByClassName("tactic");
    var i;

    for (i = 0; i < tac_spans.length; i++) {
        var cur_span = tac_spans[i];
        var region = cur_span.dataset.region;
        var idx = cur_span.dataset.index;
        const id_string = region + '-' + idx
        cur_span.id = 'command-' + id_string
        cur_span.addEventListener("mouseover", function() {
            hoverTactic(id_string)
        });
        cur_span.addEventListener("mouseout", unhoverTactic)
        cur_span.addEventListener("click", function() {
            selectTactic(id_string);
            event.stopPropagation();
        })
    }
}
function addPopupFrame() {
    var overlay = document.createElement('div');
    overlay.id = "overlay";
    overlay.addEventListener("onclick", function() { event.stopPropagation(); });
    document.body.prepend(overlay)

    var prediction_div = document.createElement('div');
    prediction_div.id = "predicted";
    overlay.appendChild(prediction_div)

    var context_div = document.createElement('div');
    context_div.id = 'context'
    overlay.appendChild(context_div)
}
function init() {
    setSelectedIdx()
    initRegions()
    addListeners()
    addPopupFrame()
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

    predictedDiv = document.getElementById("predicted")
    linkLoc = window.location.protocol + "//" + window.location.hostname
        + window.location.pathname + "#command-" + idx
    predictedDiv.innerHTML =
        "<h3>Predicted <a onclick='window.location.hash = \"\"' href=" + linkLoc + ">[link]</a></h3>"

    var actualDistancePre = document.createElement("pre")
    actualDistancePre.appendChild(document.createTextNode("Actual distance: " + tacSpan.dataset.actualDistance))
    actualDistancePre.classList.add('distance')
    predictedDiv.appendChild(actualDistancePre)
    var predictedDistancePre = document.createElement("pre")
    predictedDistancePre.appendChild(document.createTextNode("Predicted distance: " + tacSpan.dataset.predictedDistance))
    predictedDistancePre.classList.add('distance')
    predictedDiv.appendChild(predictedDistancePre)

    hyps = tacSpan.dataset.hyps//.replace(/\\n/g, "\n")
    goal = tacSpan.dataset.goal
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
