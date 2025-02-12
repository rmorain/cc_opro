from eval_utils import is_artifact_online

joke0 = (
    "Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!"
)
joke1 = "Why did the chicken cross the road? To get to the other side."
joke2 = "asl;kfjl;aisjgl;kajs;glkja;lkfgj;laksdhfiuwyqiruhvbn   jlakjds"
joke3 = 'a man walked into a library and asked the librarian, "do your books ever get lonely?" the librarian replied, "actually we\'ve been trying to shelf them."'

domain = "joke"

if is_artifact_online(joke0, domain) is True:
    print("Joke 0 is online")

if is_artifact_online(joke1, domain) is True:
    print("Joke 1 is online")

if is_artifact_online(joke2, domain) is False:
    print("Joke 2 is not online")
if is_artifact_online(joke3, domain) is False:
    print("Joke 3 is not online")
