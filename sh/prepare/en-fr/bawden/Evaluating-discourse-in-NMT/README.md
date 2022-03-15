# Evaluating-discourse-in-NMT



## Prepare training and dev data from OpenSubtitles2016

Fan-based subtitles from http://www.opensubtitles.org/  
Uses the OpenSubtitles parallel corpus:  
> Pierre Lison and Jörg Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)

To prepare OpenSubtitles2016 data:
1. cd `script-opensubs`
2. Change the paths and parameters in the prepare_data.sh file
3. Then ``bash prepare_data.sh``
