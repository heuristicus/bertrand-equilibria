set xrange [0:$0]
set title "Daily Prices"
set xlabel "Day"
set ylabel "Price (pence)"
set key below
plot "logs/price.dat" using 1:2 with lines title "Manufacturer 0",\
"logs/price.dat" using 1:3 with lines title "Manufacturer 1"
pause 1
reread
