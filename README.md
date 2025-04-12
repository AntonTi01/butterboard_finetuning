# 📘 Dataset of Interview Transcripts

This dataset consists of transcribed and diarized fragments from real interviews collected from YouTube. The dialogues were processed to assign speaker roles (`interviewer` and `candidate`), chunked, and annotated using an LLM-based assistant to extract soft skills, hard skills, and personalized recommendations.

## 🧩 Structure
Each sample in the dataset contains:
- **instruction** — guiding task description for the model.
- **input** — 4-message dialogue chunk between interviewer and candidate.
- **output** — dictionary with:
  - `hard_skills`
  - `soft_skills`
  - `recommendations`
- **chunk_id** — index of the dialogue chunk.
- **source_file** — name of the original `.json` source.

## 📥 Sources of Dialogue
| YouTube Link | Timestamps | Duration |
|--------------|------------|----------|
| [Interview 1](https://www.youtube.com/watch?v=FAA_aC8wEv8) | 06:30 – 51:00 | **44 min 30 sec** |
| [Interview 2](https://www.youtube.com/watch?v=CSrq4Y7uyfA) | 01:54 – 26:00 | **24 min 6 sec** |
| [Interview 3](https://www.youtube.com/watch?v=k2bvwkVPmS0) | 04:42 – 20:00 | **15 min 18 sec** |
| [Interview 4](https://www.youtube.com/watch?v=InDDq7azir8) | 22:30 – 1:21:00 | **58 min 30 sec** |
| [Interview 5](https://www.youtube.com/watch?v=YF5x3huyilc) | 08:31 – 30:15, 35:00 – 1:01:03 | **47 min 47 sec** |
| [Interview 6](https://www.youtube.com/watch?v=SbMFfXfErSY) | 00:00 – 1:01:50 | **61 min 50 sec** |
| [Interview 7](https://www.youtube.com/watch?v=A0hEb5IGfPs) | 07:15 – 34:05 | **26 min 50 sec** |

## 🕒 Total Duration

**278 minutes and 51 seconds**  
(≈ **4 hours and 39 minutes** of transcribed interviews)