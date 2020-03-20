ls $1/*.html | sed "s/$1\///" | sed 's/Zs/\//' | sed 's/Zdv.html/.v/'
