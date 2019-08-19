# A script to get results for alexnet

# Some constants
GREEN="\033[1;32m"
NC="\033[0m" # No Color
BLUE="\033[1;34m"


helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB -c parameterC -d parameterD"
   echo -e "\t-a Path to 92 image set"
   echo -e "\t-b Path to 118 image set"
   echo -e "\t-c Path to the model without foveation"
   echo -e "\t-c Path to the model with foveation"   
   exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:" opt
do
   case "$opt" in
      a ) parameterA="$OPTARG" ;;
      b ) parameterB="$OPTARG" ;;
      c ) parameterC="$OPTARG" ;;
      d ) parameterD="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ] || [ -z "$parameterD" ]
then
   echo "Some or all of the parameters are empty";
   echo "Path to 92 image set: $parameterA"
   echo "Path to 118 image set: $parameterB"
   echo "Path to the model without foveation: $parameterC"
   echo "Path to the model with foveation: $parameterD"

   helpFunction
fi

# Begin script in case all parameters are correct
echo "Path to 92 image set: $parameterA"
echo "Path to 118 image set: $parameterB"
echo "Path to the model without foveation: $parameterC"
echo "Path to the model with foveation: $parameterD"

# Clean the directories for a fresh start
# Delete the feats and the rdms folder
printf "${BLUE}Info:${NC} Cleaning the directories\n"
if [ -d "Feature_Extract/feats" ]; then rm -Rf Feature_Extract/feats; fi
if [ -d "Feature_Extract/rdms" ]; then rm -Rf Feature_Extract/rdms; fi
printf "${GREEN}OK:${NC} Cleared\n"

# Run the code
cd Feature_Extract
# Create features
printf "${BLUE}Info:${NC} Creating features for 118 image set with foveation\n"
python3 generate_features.py --image_dir  $parameterA --save_dir  "./feats"  --net alexnet --load_model $parameterD --exp alex_foveate
printf "${GREEN}OK:${NC} Features created for 118 image set with foveation\n"
printf "${BLUE}Info:${NC} Creating features for 118 image set without foveation\n"
python3 generate_features.py --image_dir  $parameterA --save_dir  "./feats"  --net alexnet --load_model $parameterC --exp alex_no_foveate
printf "${GREEN}OK:${NC} Features created for 118 image set without foveation\n"

printf "${BLUE}Info:${NC} Creating features for 92 image set with fovetion\n"
python3 generate_features.py --image_dir  $parameterB --save_dir  "./feats"  --net alexnet --load_model $parameterD --exp alex_foveate
printf "${GREEN}OK:${NC} Features created for 92 image set with foveation\n"
printf "${BLUE}Info:${NC} Creating features for 92 image set without fovetion\n"
python3 generate_features.py --image_dir  $parameterB --save_dir  "./feats"  --net alexnet --load_model $parameterC --exp alex_no_foveate
printf "${GREEN}OK:${NC} Features created for 92 image set without foveation\n"

# Create RDMs
printf "${BLUE}Info:${NC} Creating RDMs for 118 image set\n"
python3 create_RDMs.py --feat_dir "./feats/118images_feats" --save_dir './rdms/118images_rdms'
printf "${GREEN}OK:${NC} RDMs created for 118 image set\n"

printf "${BLUE}Info:${NC} Creating RDMs for 92 image set\n"
python3 create_RDMs.py --feat_dir "./feats/92images_feats" --save_dir './rdms/92images_rdms'
printf "${GREEN}OK:${NC} RDMs created for 92 image set\n"

# Test for goodness
cd ../Evaluation_Scripts
printf "${BLUE}Info:${NC} Let the games begin\n"
python3 testSub_fmri.py ../Feature_Extract/rdms

# Move the results into a directory
if [ ! -d "Results" ]; then mkdir Results; fi
mv *.txt Results
mv Results/ReadMe_Evaluation.txt .
printf "${GREEN}OK:${NC} All done; hopefully\n\n\n"
