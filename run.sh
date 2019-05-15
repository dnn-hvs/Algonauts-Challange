# A file to run the entire project at once

# Some constants
GREEN="\033[1;32m"
NC="\033[0m" # No Color
BLUE="\033[1;34m"

# Clean the directories for a fresh start
clear
# Delete the feats and the rdms folder
printf "${BLUE}Info:${NC} Cleaning the directories\n"
if [ -d "Feature_Extract/feats" ]; then rm -Rf Feature_Extract/feats; fi
if [ -d "Feature_Extract/rdms" ]; then rm -Rf Feature_Extract/rdms; fi
printf "${GREEN}OK:${NC} Cleared\n"

# Run the code
cd Feature_Extract
# Create features
printf "${BLUE}Info:${NC} Creating features for 118 image set\n"
python3 generate_features.py --image_dir  "../Training_Data/118_Image_Set/118images" --save_dir  "./feats"  --net all 
printf "${GREEN}OK:${NC} Features created for 118 image set\n"

clear
printf "${BLUE}Info:${NC} Creating features for 92 image set\n"
python3 generate_features.py --image_dir  "../Training_Data/92_Image_Set/92images" --save_dir  "./feats"  --net all 
printf "${GREEN}OK:${NC} Features created for 92 image set\n"

# Create RDMs
clear
printf "${BLUE}Info:${NC} Creating RDMs for 118 image set\n"
python3 create_RDMs.py --feat_dir "./feats/118images_feats" --save_dir './rdms/118images_rdms'
printf "${GREEN}OK:${NC} RDMs created for 118 image set\n"

clear
printf "${BLUE}Info:${NC} Creating RDMs for 92 image set\n"
python3 create_RDMs.py --feat_dir "./feats/92images_feats" --save_dir './rdms/92images_rdms'
printf "${GREEN}OK:${NC} RDMs created for 92 image set\n"

# Test for goodness
clear
cd ../Evaluation_Scripts
printf "${BLUE}Info:${NC} Let the games begin\n"
python3 testSub_fmri.py ../Feature_Extract/rdms

# Move the results into a directory
if [ ! -d "Results" ]; then mkdir Results; fi
mv *.txt Results
mv Results/ReadMe_evaluation.txt .
printf "${GREEN}OK:${NC} All done; hopefully\n\n\n"
