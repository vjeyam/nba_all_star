# #-------------- Import libraries and packages --------------#
library(tidyverse)
library(dplyr)
library(ggplot2)

#-------------- Import datasets --------------#
nba_all_stars <- read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")

#-------------- Clean data / Feature engineering --------------#

#-------------- Split data into features / Target / Train / Validation --------------#

#-------------- See how the number of three pointers changed over time --------------#
three_point_evolution <- function(nba_all_stars) {
    # Group the all stars by year and find average number of 3 pointers attempted by all stars
    nba_all_stars %>%
        group_by(year) %>%
        summarize(avg_fg3a = mean(fg3a, na.rm = TRUE)) -> avg_3_attempt

    # plot the average number of three pointers attempted by all stars over the years of the data
    avg_3_attempt %>%
        ggplot(mapping = aes(x = year, y = avg_fg3a)) +
        geom_point(color = "blue") + 
        geo_smooth(color = "black", alpha = 0.25) + 
        geo_vline(xintercept = 2009, color = "gold") # adds a vertical line at 2009
}
#--------------------------------------------------------------------#

# Call functions
result_plot <- three_point_evolution()