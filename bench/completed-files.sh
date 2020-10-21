[ "$#" -eq 1 ] || (echo "1 argument required, $# provided" && exit 1)
find $1 -name "*.html" | grep -v "report.html" | sed "s~$1/~~" | sed 's/Zs/\//g' | sed 's/Zdv.html/.v/' | sed 's/ZZ/Z/g'
