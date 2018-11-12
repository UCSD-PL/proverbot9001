
window.onload = function () {
    make_rows_clickable()
    add_checkboxes()
    render_graph(all_predictors())
}

function add_checkboxes() {
    var data = all_predictors()
    var div = d3.select("div.checkbox-box")
    div.append("p")
        .attr("class", "control-label")
        .text("Predictors: ")
    for (var i = 0; i < data.length; i++) {
        div.append("input")
            .attr("type", "checkbox")
            .attr("name", data[i])
            .attr("value", data[i])
            .attr("checked", '')
            .on("change", update_checkboxes);
        div.append("p")
            .attr("class", "checkbox-label " + data[i])
            .text(data[i])
    }
}

function update_checkboxes() {
    var all_checkboxes = document.getElementsByTagName("input")
    var predictors = []
    for (var i = 0; i < all_checkboxes.length; ++i){
        if (all_checkboxes[i].checked) {
            predictors.push(all_checkboxes[i].value)
        }
    }
    render_graph(predictors)
}

function searchClass(list, classname) {
    for (var i = 0; i < list.length; i++){
        if (list[i].className == classname){
            return list[i]
        }
    }
}

function make_rows_clickable() {
    var rows = document.getElementsByTagName("tr")
    for (var i = 0; i < rows.length; i++){
        if (rows[i].className == "header") continue
        rows[i].onclick = (function (row) {
            return function () {
                window.location = searchClass(row.children, "link").children[0].href;
            }
        })(rows[i])
        rows[i].onmouseover = (function (row, idx) {
            return function () {
                var elementId = "dot" + idx;
                var dot = document.getElementById(elementId);
                if (dot == null){
                    console.log("Cannot find element with id " + elementId);
                } else {
                    dot.setAttribute("r", 12);
                }
            }
        })(rows[i], i)
        rows[i].onmouseout = (function (row, idx) {
            return function () {
                var elementId = "dot" + idx;
                var dot = document.getElementById(elementId);
                if (dot == null){
                    console.log("Cannot find element with id " + elementId);
                } else {
                    dot.setAttribute("r", 8);
                }
            }
        })(rows[i], i)
    }
}
function all_predictors() {
    var rows = document.getElementsByTagName("tr")
    var predictors_seen = []
    for (var i = 0; i < rows.length; i++){
        if (rows[i].className == "header") continue
        var predictor = searchClass(rows[i].children, "predictor").innerText;
        predictors_seen.push(predictor)
    }
    return predictors_seen.filter(function(item, pos, self) {
        return self.indexOf(item) == pos;
    })
}

function render_graph(predictors) {
    var svg = d3.select("svg");
    svg.selectAll("g").remove()
    var margin = {top: 20, right: 50, bottom: 50, left: 50},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom,
        g = svg.append("g").attr("transform",
                                 "translate(" + margin.left + "," + margin.top + ")");

    var parseTime = d3.timeParse("%a %b %d %Y %H:%M");
    var x = d3.scaleTime()
        .rangeRound([0, width]);
    var y = d3.scaleLinear()
        .rangeRound([height, 0]);

    var line = d3.line()
        .x(function(d) { return x(d.date); })
        .y(function(d) { return y(d.percent_correct)});

    var data = {}
    var all_data = []
    var rows = document.getElementsByTagName("tr")
    for (var i = 0; i < rows.length; i++){
        if (rows[i].className == "header") continue;
        var d = {};
        d.percent_correct =
            +(/(\d+\.\d+)%/.exec(searchClass(rows[i].children, "accuracy").innerText)[1]);
        var lookback = 0
        while(rows[i-lookback].children[0].className != "date"){
            lookback += 1
        }
        var datetext = rows[i-lookback].children[0].innerText;
        var timetext = searchClass(rows[i].children, "time").innerText
        d.date = parseTime(datetext + " " + timetext);
        d.idx = i;
        var predictor = searchClass(rows[i].children, "predictor").innerText;
        if (predictors.indexOf(predictor) < 0) continue;
        all_data.push(d)
        if (typeof data[predictor] == "undefined"){
            data[predictor] = []
        }
        data[predictor].push(d);
        all_data.push(d)
    }

    x.domain(d3.extent(all_data, function(d) { return d.date; }));
    y.domain(d3.extent(all_data, function(d) { return d.percent_correct; }));

    var axis = g.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).tickFormat(d3.timeFormat("%Y-%m-%d")))
        .attr("font-size", 16);
    axis.selectAll("text")
        .attr("transform", "rotate(-65)");
    axis.select(".domain")
        .remove();

    g.append("g")
        .call(d3.axisLeft(y))
        .attr("font-size", 16)
        .append("text")
        .attr("fill", "#000")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .attr("font-size", 20)
        .text("Percentage Correct");

    predictors = Object.keys(data)

    for (var i = 0; i < predictors.length; ++i) {
        key = predictors[i]
        g.append("path")
            .datum(data[key])
            .attr("fill", "none")
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
            .attr("stroke-width", 4)
            .attr("d", line)
            .attr("class", "graphline " + key);

        g.selectAll(".dot ." + key)
            .data(data[key])
            .enter()
            .append("circle")
            .attr("class", "dot " + key)
            .attr("r", 8)
            .attr("cx", function(d){
                return x(d.date);
            })
            .attr("cy", function(d){
                return y(d.percent_correct);
            })
            .attr("id", function(d){
                return "dot" + d.idx;
            })
            .on("mouseover", function(d) {
                var row = document.getElementsByTagName("tr")[d.idx];
                d3.select(this).attr("r", 12);
                row.classList.add("highlighted");
            })
            .on("mouseout", function(d) {
                var row = document.getElementsByTagName("tr")[d.idx];
                d3.select(this).attr("r", 8);
                row.classList.remove("highlighted");
            })
            .on("click", function(d) {
                var row = document.getElementsByTagName("tr")[d.idx];
                window.location = searchClass(row.children, "link").children[0].href;
            });
    }
}
