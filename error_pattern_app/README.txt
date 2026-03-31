# ErrPattern — Student Error Clustering System

## Setup (run once)

1. Open Command Prompt in this folder
2. Install dependencies:
   pip install -r requirements.txt

## Run the app

   python app.py

Then open your browser and go to:
   http://localhost:5000

## How to use

1. UPLOAD CSV — your file must have: student_id, q1, q2, ... q30
2. SET K — number of clusters (default: 3)
3. ADD DOMAINS — set domain names and which item numbers belong to each
4. ADD DISTRACTOR CATEGORIES — the error type labels you used (e.g. off_by_one)
5. GENERATE ANSWER KEY — click the button, then set each choice per item
6. RUN ANALYSIS — see all results, charts, and validation scores

## CSV format example

student_id,q1,q2,q3,...,q30
S001,A,C,B,...,D
S002,C,A,A,...,C

## Notes
- Domains and distractor categories are fully customizable
- Works for any subject, not just Math in the Modern World
- All charts and tables update automatically after each run
