
window.onload = function () {
    var rows = document.getElementsByTagName("tr")
    for (var i = 0; i < rows.length; i++){
        if (rows[i].className == "header") continue
        rows[i].onclick = (function (row) {
            return function () {
                window.location = row.children[2].children[0].href
            }
        })(rows[i])
    }
}
