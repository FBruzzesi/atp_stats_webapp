# Tennis Analytics
Webapp built in Python Dash for tennis analytics.

**Data Attribution:** The data used here is (part of) the amazing dataset created by [**Jeff Sackmann**](http://www.jeffsackmann.com/) 
(Check out his [github repository](https://github.com/JeffSackmann/tennis_atp))

**Data Usage:** In particular, I am using atp tour-level main draw single matches from 1995 to present day. I am currently working towards an independent data gathering solution.

**Bug Fix:** This is a MVP which I had fun developing, mostly on weekends, for personal use. Therefore I am sure it is possible to find bugs and non-working interactions. 
If you find any or just want to get in touch with me, please feel free to reach out by [Linkedin](https://www.linkedin.com/in/francesco-bruzzesi/)

**Support:** I would love to grow the project, if you feel like supporting, you can [buy me a coffee ☕](https://www.buymeacoffee.com/fbruzzesi)

**How it Works:** Down below there are a series of possible filters you want to play with. Everything is based upon a selected player, in the sense that only such player statistics will appear. Then:

- _Player Summary_: Shows rank, rank points, winrate over time and a set of overall statistics as well as some player information.
- _Serve & Return_: Shows serve and return statistics over time with a 95% confidence interval and distribution of all selected matches.
- _Under Pressure_: Shows under pressure statistics over time with a 95% confidence interval.
- _H2H_: Shows winrate againsts most played opponents.