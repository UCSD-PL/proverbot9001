
window.onload = function () {
    make_rows_clickable()
    render_graph()
}

function make_rows_clickable() {
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

function render_graph() {
    var svg = d3.select("svg"),
        margin = {top: 20, right: 20, bottom: 30, left: 50},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom,
        g = svg.append("g").attr("transform",
                                 "translate(" + margin.left + "," + margin.top + ")");

    var parseTime = d3.timeParse("%a %b %e %X %Y");
    var x = d3.scaleTime()
        .rangeRound([0, width]);
    var y = d3.scaleLinear()
        .rangeRound([height, 0]);

    var line = d3.line()
        .x(function(d) { return x(d.date); })
        .y(function(d) { return y(d.percent_correct)});

    var data = []
    var rows = document.getElementsByTagName("tr")
    for (var i = 0; i < rows.length; i++){
        if (rows[i].className == "header") continue
        var d = {};
        d.percent_correct = /(\d+\.\d+)%/.exec(rows[i].children[1].innerText)[1]
        d.date = parseTime(rows[i].children[0].innerText)
        data.push(d)
    }

    x.domain(d3.extent(data, function(d) { return d.date; }));
    y.domain(d3.extend(data, function(d) { return d.close; }));

    g.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .select(".domain")
        .remove();

    g.append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("fill", "#000")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .text("Percentage Correct");

    g.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("d", line);
}
