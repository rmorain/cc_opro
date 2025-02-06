from eval_utils import is_joke_online

joke1 = "Why did the chicken cross the road? To get to the other side."
joke2 = "asl;kfjl;aisjgl;kajs;glkja;lkfgj;laksdhfiuwyqiruhvbn   jlakjds"
joke3 = 'a man walked into a library and asked the librarian, "do your books ever get lonely?" the librarian replied, "actually we\'ve been trying to shelf them."'

if assert is_joke_online(joke1) is True:
    print("Joke 1 is online")

if assert is_joke_online(joke2) is False:
    print("Joke 2 is not online")
if assert is_joke_online(joke3) is False:
    print("Joke 3 is not online")
