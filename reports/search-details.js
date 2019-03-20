
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
    overlay.setAttribute("id", "overlay")
    overlay.classList.add("graph")
    overlay.style.display = "block"
    document.body.appendChild(overlay)
    image = document.createElement("img")
    image.classList.add("search-graph")
    image.src = lemma_name + ".png"
    overlay.appendChild(image)
}
function unhoverLemma(lemma_name) {
    existing_overlay = document.getElementById("overlay")
    if (existing_overlay != null){
        document.body.removeChild(existing_overlay)
    }
}
