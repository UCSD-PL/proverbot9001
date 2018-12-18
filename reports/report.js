
window.onload = function () {
    make_rows_clickable()
}
function make_rows_clickable() {
    var rows = document.getElementsByTagName("tr")
    for (var i = 1; i < rows.length; i++){
        if (rows[i].className == "header") continue
        rows[i].onclick = (function (row) {
            return function () {
                window.location = row.children[row.children.length-1].children[0].href
            }
        })(rows[i])
        rows[i].style.cursor = "pointer"
    }
}
