#!/usr/bin/env gnuplot
set xrange [-200:200]
set yrange [-200:200]
plot 'tmp_trajectory.txt' u 1:2 lc rgb 'red', 'tmp_trajectory.txt' u 3:4 lc rgb 'blue', 'tmp_trajectory.txt' u 5:6 lc rgb 'green', 'tmp_trajectory.txt' u 7:8 lc rgb 'yellow'
while (1) { pause 1; replot; }
