
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
function fromList(list_str) {
    trimmed = list_str.trim()
    valid = trimmed[0] == "(" && trimmed[trimmed.length-1] == ")"
    console.assert(valid)
    body = trimmed.slice(1, -1)
    if (body == ""){
        return []
    }
    in_quotes = 0
    paren_depth = 0
    items = [""]
    var i;
    for(i = 0; i < body.length; ++i){
        c = body[i]
        if (c == "\"" && body[i-1] != "\\"){
            in_quotes = !in_quotes
        } else if (c == "(" && !in_quotes){
            paren_depth += 1
        } else if (c == ")" && !in_quotes){
            paren_depth -= 1
        }
        if (c == "," && !in_quotes && paren_depth == 0){
            items.push("")
        } else {
            items[items.length-1] += c
        }
    }
    // result = body.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/g)
    return items
}
function parseSubgoals(subgoals_str) {
    subgoal_strs = fromList(subgoals_str)
    subgoals = []
    var i;
    for (i = 0; i < subgoal_strs.length; i++) {
        subgoal = {}
        parts = fromList(subgoal_strs[i])
        subgoal.goal = parts[0]
        subgoal.hypotheses = fromList(parts[1]).map(hyp => hyp.slice(1, -1))
        subgoals.push(subgoal)
    }
    return subgoals
}
function hoverTactic(idx_str){
    overlay = document.createElement('div')
    overlay.setAttribute("id", "tactic_overlay")
    overlay.classList.add("context")
    overlay.style.display = "block"
    document.body.appendChild(overlay)

    tacSpan = document.getElementById("command-" + idx_str)
    subgoals = parseSubgoals(tacSpan.dataset.subgoals)

    var i;
    for (i = 0; i < subgoals.length; i++) {
        subgoal = subgoals[i]
        hypsPre = document.createElement('pre')
        hypsPre.setAttribute("id", "hyps")
        hypsPre.innerText = subgoal.hypotheses.join("\n")
        overlay.appendChild(hypsPre)

        rule = document.createElement('hr')
        overlay.appendChild(rule)

        goalPre = document.createElement('pre')
        goalPre.setAttribute('id', 'goal')
        goalPre.innerText = subgoal.goal.slice(1, -1)
        overlay.appendChild(goalPre)
        br = document.createElement('br')
        overlay.appendChild(br)
        br2 = document.createElement('br')
        overlay.appendChild(br2)
    }
}
function unhoverTactic() {
    existing_overlay = document.getElementById("tactic_overlay")
    if (existing_overlay != null){
        document.body.removeChild(existing_overlay)
    }
}
