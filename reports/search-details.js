
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
    initRegions()
}

function hoverLemma(lemma_name){
    overlay = document.createElement('div')
    overlay.setAttribute("id", "lemma_overlay")
    overlay.classList.add("graph")
    overlay.style.display = "block"
    document.body.appendChild(overlay)
    image = document.createElement("img")
    image.classList.add("search-graph")
    image.src = lemma_name + ".png"
    overlay.appendChild(image)
}
function unhoverLemma(lemma_name) {
    existing_overlay = document.getElementById("lemma_overlay")
    if (existing_overlay != null){
        document.body.removeChild(existing_overlay)
    }
}
function hoverTactic(idx_str){
    overlay = document.createElement('div')
    overlay.setAttribute("id", "tactic_overlay")
    overlay.classList.add("context")
    overlay.style.display = "block"
    document.body.appendChild(overlay)

    tacSpan = document.getElementById("command-" + idx_str)
    hyps = tacSpan.dataset.hyps
    goal = tacSpan.dataset.goal

    hypsPre = document.createElement('pre')
    hypsPre.setAttribute("id", "hyps")
    hypsPre.innerText = hyps
    overlay.appendChild(hypsPre)

    rule = document.createElement('hr')
    overlay.appendChild(rule)

    goalPre = document.createElement('pre')
    goalPre.setAttribute('id', 'goal')
    goalPre.innerText = goal
    overlay.appendChild(goalPre)
}
function unhoverTactic() {
    existing_overlay = document.getElementById("tactic_overlay")
    if (existing_overlay != null){
        document.body.removeChild(existing_overlay)
    }
}
