# Daily Crime Data - LAPD - 2020 Forward
This repository contains the code script for a pull of daily crime data as reported by the LAPD through their open source API - reported here: https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8/about_data
The code includes the pull request as well as some initial data cleansing, stadardization, and classification summaries. As an example, the victim descent field as been expaned from a letter - such as A - to its full description which in this case would be Other Asian. The sex of the victim has also been expanded to Male and Female from M and F. Crime Codes have also been mapped from the first level (i.e., Crim_Cd_1 to their corresponding code description). You will need to use the 'lacrimedatafifteencat' file in conjunction with the code script to have the same mapped dictionary referenced in the python code. The weapons used and the premis of where the crime occured have also been mapped up to a higher summarized level (i.e., Firearms, Edged and Blunt Weapons, etc. and Public Spaces and Transit, Mass Transit Locations, and Financial Institutions, etc.). This was done with the assistance of ChatGPT providing dialgoue related to how best to summarize (roll up) the numerous values included in the original dataset's premis description and weapon description. This allows data aggregation for visual exploration and illustration of the data as well as easier statistical and catergorical analysis.
