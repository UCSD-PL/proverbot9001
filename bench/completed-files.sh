ls $1/*.html | sed "s/$1\///" | sed 's/Zs/\//g' | sed 's/Zdv.html/.v/' | sed 's/ZZ/Z/g'
