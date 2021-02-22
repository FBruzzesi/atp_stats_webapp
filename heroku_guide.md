# What is Heroku?
Heroku is a platform-as-a-service (PaaS) for deploying and hosting web applications. In the context of a Dash app, this means Heroku provides the physical hardware (storage, compute), software services (linux/unix/sql), and dependencies (packages), in a containerised environment to deploy and host your application on a publicly accessible URL (end-point). It does this through provision of virtualised linux containers called “Dynos” which essentially act as your personal linux webserver, with ram, cpu, and linux ‘installed’. (It’s not quite like this in reality but a good analogy).
Dynos come in a variety of types and can be scaled vertically (more ram, compute, storage per instance) or horizontally (duplicate dynos in parallel) as your specific project requirements demand. This can be done almost instantaneously at command line. The free version gets you one dyno with up to 500MB storage and 500MB ram. It sleeps after 30 minutes of inactivity, presumably so Heroku resources are not drained. So the catch with the free version is that your website can take a good 30–60 seconds to load initially, as your free Dyno is provisioned on demand. If you go to a paid plan, starting at about $7USD/month, your dyno(s) stay on and ready 24hrs/day.

# Why GitHub?
In short — Heroku natively supports deploying repositories that reside in GitHub. This is good news. Basically it means if your project is already in a GitHub repository (free for private/public repos) then you can easily deploy it on Heroku _after_ you have added a few additional files that are outlined in the deployment guides, and in a section further below.

# What is Gunicorn?
[Gunicorn](https://gunicorn.org/) is a production-ready HTTP server specifically for Python web applications which runs natively in Unix. If you’ve been developing your Dash app purely on your local machine @localhost:8080 or http://127.0.0.1:8050/ you will be running a light weight HTTP server that is shipped with your Python installation. This is not Gunicorn. It’s likely you have not yet glimpsed this rare and mythical creature of the forest.

The local HTTP server (shipped with your Python installation) is automatically run by your Python Kernel when your dash app is executed on your local machine. The issue is that it’s not designed for handling incoming traffic from a production website and so when you deploy to the web, you need a production-ready HTTP server. A popular one is Gunicorn. Notably, Heroku provides native support for Gunicorn, which makes things easy. It’s all outlined in the guides, but just to clarify, all you need to do is add a single line of code to your dash app (`server = app.server`), add Gunicorn into your requirements.txt so it is installed as a package on your local machine (and by Heroku during deployment), and reference it in a special file you will create called the Procfile. More on this later but I think it’s worth briefly touching on the HTTP server as it’s all a bit mysterious the first time.

# Web is hard
This is a simple truth. Web is multi-layer, multi-language, multi-protocol, multi-platform, and multi-user. It’s a mind-boggling chain of infrastructure bolted to other infrastructure to make a modern (scalable) web application run. For many non-IT people (and IT people for that matter), even the concept of a locally hosted webserver takes a bit of abstract thought, let alone understanding the true technology stack that lies underneath a real client-server architecture. It’s also worth reflecting on just how new some of this technology really is.

# Dash-Heroku deployment, in a nutshell
What actually needs to be done:
1. Dash app running on localhost.
2. Install Git.
3. Setup GitHub account (+ recommend install GitHub Desktop).
4. Setup Heroku account (+ install the command line interface).
5. Add dependencies and special files (i.e. install and import Gunicorn, create Procfile and runtime.txt).
6. Clone repo from GitHub to local machine (only once).
7. Create Heroku app linked to your repo (only once, ref deployment guides, Heroku CLI).
8. Commit and push your code changes to GitHub repo (repetitively).
9. Deploy/Re-deploy Heroku app by pushing changes from Heroku CLI (`git push Heroku main` or `git push heroku HEAD:master`).

# Deployment Guides
The guides below are concise and useful, and I would of course start with these. If I’m honest, I think they are a little light on detail for newcomers and would benefit greatly by having a supplementary explanatory guide akin to something like this essay.
- [Plotly’s Dash deployment guide](https://dash.plotly.com/deployment)
- [Heroku’s guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Install Heroku command line interface (CLI)](https://devcenter.heroku.com/articles/heroku-cli)
- Also, this [YouTube tutorial](https://www.youtube.com/watch?v=b-M2KQ6_bM4&feature=youtu.be&ab_channel=CharmingData) from a fellow Plotly Community Forum member.

# The magical ingredients to add to your project
A quick note on the special files you need uniquely to get your python project deployed to Heroku. This is outlined in the deployment guide, so I’ve just provided a few notes from my experience:
## Ingredient 1: Procfile
This strange extensionless file must reside in your project root, and tells Heroku how to handle web processes (in our case using Gunicorn HTTP server) and the name of your Python application. Typically the Procfile would contain a single line: `web: gunicorn app:server`

Where:
- `web:` tells Heroku the dyno main process is a web process.
- `gunicorn` tells heroku that the HTTP server to use is Gunicorn (for which it has native support for).
- `app` references the filename of the main python file without the .py extension. So if you follow the convention of ‘app.py’ you would use ‘app’ here. But note if your main python file is ‘anything.py’, you would have ‘anything’ in place of ‘app’.
- `server` references the underlying flask app. Commonly you would define a variable ‘server = app.server’ and this references that variable (I believe). To be more confusing, the ‘app’ in this variable declaration actually refers to the dash instantiation variable i.e. 
```
app = dash.Dash(__name__)
server = app.server
```

Yes I know what you’re thinking, this is finicky and it’s really easy to misunderstand with all these ‘app’ references everywhere. Take home is: as long as you’re using an app.py main file, as is the convention, and you declare a ‘server = app.server’ line of code after your Dash declaration, you can use the example Procfile and it should work. If you get anything with the Procfile wrong, pain and suffering will ensue.

To make the Procfile in Windows, you can just create a text file, and enter the single line. Then strip out the extension. This worked for me and I don’t need to have a secondary Procfile.win, which is sometimes talked about in the documentation.

## Ingredient 2: runtime.txt
This file (which must also be in your root project folder) simply tells Heroku which Python runtime to use. Currently it can contain a single line, e.g.: `python-3.7.8`.

Just create this as a notepad .txt file in your project folder, and commit-push to your remote GitHub repo. Done.
That’s really it. It’s mainly these two files (Procfile, runtime.txt) that Heroku needs in your repo project directory in order to work. As long as you have followed the basics, and added Gunicorn to your requirements.txt etc, in theory you are good to go.

## Ingredient 3: perseverance
Not to be underestimated! Dogged, stubborn perseverance is a key ingredient to the potion.

# Time for magic
You’ve got your code in a GitHub repository, with the required tweaks and files created. You have Heroku CLI installed and have created a Heroku app linked to your GitHub repo. It’s 4am and the sun is coming up soon. The scene is grim with pizza boxes on the floor and red wine stains on the keyboard. It’s show time.

Cast the magic spell: `git push heroku main`

These four words are the spell that makes the magic happen. Type them into the Heroku CLI in the right conditions, sit back smugly, and enjoy the show.

For those new to Heroku, if everything has worked after your `git push Heroku main` from the Heroku CLI, your app will be deployed to a Heroku subdomain like: http://blah.herokuapp.com/

If this is the case, recommend a little dance: copy-paste the URL displayed in the Heroku CLI into your browser and get ready to... be disappointed. Chances are the first time you will see **APPLICATION ERROR** in the browser or something like that. _Don’t panic_.

The first thing you should do is bring up the log (which is effectively your python console) and see what’s going on, from the Heroku CLI. Any print statements, or logger outputs from your code will display here just as they do in the console on your local machine. But most importantly, you also see Heroku system outputs such as Dyno restarts, crashes, and error messages. This is the critical way to see what’s happening with your app at a low level. I usually keep a dedicated CMD window open running Heroku logs, which is my console output.

View logs: `heroku logs --tail`

Check for things like _Module not found errors_ and simple things like that. The most common problems I’ve found are forgetting to add packages to my requirements.txt file because I frantically installed them to my local machine with conda/pip to get something working. If you’ve found some obvious problems, fix them, commit-push your code to GitHub, and then redeploy from Heroku CLI with `git push heroku main`.

Notably though, the first time is indeed the hardest. Errors in your Procfile, for example, can still cause Heroku to deploy successfully, but the dynos will crash or fail to start, so definitely check the Procfile.

Special note if there are no changes to your remote repo on GitHub, Heroku will not deploy. (Which makes sense.) So if you have a GitHub repo cloned onto your local machine, and you are making changes, be sure to commit and push changes back to your remote GitHub repo first (either with command or with GitHub desktop), then in the Heroku CLI terminal, just type in the `git push heroku main` deploy command.

## HEROKU TIP: Useful commands from Heroku CLI
Below is a cheat-sheet of important tips, pitfalls, and pitfall solutions when using Heroku.

- Explicitly referencing your app name:

    Note that Heroku can sometimes be funny about requiring you to explicitly specify your app in the command. If you just have a single Heroku app, often you can avoid it. But sometimes (without any apparent reason) you may need to append `-a <yourapp>` to the Heroku command.

- Display current apps: `heroku apps`
- Display current dynos: `heroku ps`, `heroku ps -a <yourapp>`
- [Scale dynos](https://devcenter.heroku.com/articles/scaling): `heroku ps:scale web=2:standard-1x` 
    
    In this case, we are provisioning two standard-1x dynos to run concurrently. Special note, if WEB_CONCURRENCY=4, this means each Dyno can serve 4 simultaneous HTTP incoming requests, meaning your whole Dash application can serve 8 concurrent requests — the benefit of horizontal scaling. More on this later.

- Run bash terminal: `heroku run bash -a <yourapp>`
- Restart dynos: `heroku dyno:restart`
- Add additional log metrics: `heroku labs:enable log-runtime-metrics`
- View logs: `heroku logs --tail`

## HEROKU TIP: Add runtime metrics to log
From the Heroku CLI (once logged in) when you have deployed your app, you can view a live log tail by typing `heroku logs --tail`.

Repeated just in case you missed it. This is mission critical. It essentially gives you your console output.

 One thing I'd suggest is adding in a new feature that outputs resources statistics of your dyno(s) timestamped every ~20 seconds, like memory levels, cpu load etc, which is very useful. 

Type this in the Heroku CLI, to permanently add it: `heroku labs:enable log-runtime-metrics`

## HEROKU PITFALL: Serving static files does not work
I repeat: serving static files DOES NOT WORK. Something of paramount importance that is not obvious, is that out-of-the-box, Heroku (I think more correctly: Gunicorn itself) does not natively support serving static files. This means that while your python application itself can access files in any subfolder in your project folder (such as .csv files etc.), it’s a very different story to actually serve them via http in the client browser.

Any images, documents, video, audio, anything you are currently serving from your ‘localhost’ webserver will fail on deployment with Heroku. I believe this is a quirk of the PaaS model in that files themselves are not stored in the traditional way you would imagine them to be on a file system, so there are issues with low level connection/packet headers that are attached to files, and/or Gunicorn itself does not natively support serving static files. In any case, there’s magic under the hood.

As an aside, if you don’t already know from the docs, it’s important to understand that the Heroku file system is not persistent. Like many of my past relationships, Heroku’s file system is ephemeral or transient; it lasts about as long as a one-night stand. With the exception of the files you deploy with your project repo (e.g. csv, json files etc.), any new files created at runtime will disappear after a few days.

Anyway, to store and serve persistent static files, as I said, any files _uploaded to Heroku_ as part of your project file suite will be fine, persistent, and accessible by your dash app internally. 

BUT the moment you want to serve static files externally to browser, you will rapidly run into problems. There are two main solutions I know of. One is simple and fast.

Solutions:
- Host your files on a 3rd party like S3, Cloudfront and link the URL in your dash app (Worth doing if you will be hosting a serious footprint of files).
- Use the [Whitenoise](http://whitenoise.evans.io/en/stable/) library. Quick and easy. A few lines of code and you’re serving files just like in your localhost setup.

Personally I found Whitenoise to be a life saver. Literally `pip install whitenoise` (and make sure it’s in your requirements.txt) and you’re almost there. 

A few lines of code needed in your dash app: 

```python
from whitenoise import WhiteNoise
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')
```
You should already have the `server=app.server` anyway, as this is needed by Gunicorn and for the Procfile.

What this essentially does is wrap Whitenoise around your underlying Flask app. You can then have a folder (which you must create) called `static/` in your root. Everything contained within this (including subfolders) can be statically served by Heroku. Images, videos, pdfs, whatever you want.

Special note Heroku is file extension case-sensitive. So blah.png is different to blah.PNG.
Also, don’t try to get smart and change the 'static' folder name in the Whitenoise code declaration to some arbitrary name or 'assets' or anything like that: it must be 'static' due to an underlying Flask constraint.

This seems like a pretty major issue that I don’t think much documentation exists on. I spent a long time on Stack Overflow looking it up.

Also, the Whitenoise documentation is not specific to Dash — it is more focused on general Python apps which are typically Flask apps. This means that it’s still not obvious what you need to do, and the code snippets will not work without modification. For example Whitenoise states that, for Flask apps, you must add the following code to your app:
```python
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
```

This won’t work for your dash app. In this case ‘app’ is the Flask app. So in a Dash app (which sits on top of Flask) you actually need to replace the ‘app’ reference with ‘app.server’ in the snippet above to reference the underlying Flask app and for Whitenoise to work. Or simply define a variable such as ‘server = app.server’ and use the code snippet I outlined at the beginning of this section.

Again, lots of these things are a two second fix if you know how. But can cost you literally _hours and hours and ... hours_ of time. Trivial for Flask developers, not trivial at all for newcomers.

## HEROKU PITFALL: Favicon may not work
For some reason, I had lots of trouble with this. Anyway, I managed to get it going by simply having a: `/assets/favicon.ico`

From my root project directory. Special note that no other static files are served from here; it’s a stand-alone folder. In fact, don’t be lulled into thinking you can serve static files from your /assets folder on Heroku: you can’t. (see Whitenoise section).

Others have had problems with Heroku changing the extension name of the favicon causing it to fail. One failsafe option to note is that you can log into a Heroku Bash shell after you have deployed, and navigate to all your project folders/files to see what Heroku sees. See [this post](https://community.plotly.com/t/display-favicon-in-heroku-dash-app/44013/3).

From heroku CLI: `heroku run bash -a <yourappname>`

This will provision a new Dyno container running a Bash shell. Basically, it’s a terminal to your deployed app.

## HEROKU PITFALL: Web concurrency is important and can be configured
There is lots of ‘worker’ and ‘web’ terminology that gets confusing. Out of the box when using Gunicorn as your Python HTTP server, Heroku essentially guesses how many concurrent web-worker-processes to run for each dyno instance running your web app. Typically this is 1–6 concurrent ‘gunicorn — web-worker-processes’ per dyno for the commonly used hobby to standard 2-x dynos. This is how many client requests (i.e. from a web browser) can be simultaneously served by your app at an instantaneous point in time.

A Gunicorn web-worker-process is a process capable of serving a single HTTP request at a time. So if you only had one, this means your website becomes quite unresponsive with a few users making simultaneous requests, and having to wait for these requests to be actioned from a queue. 

Essentially this is the magic of what Gunicorn does: it forks the main web process running on its Dyno into multiple processes so that it can serve simultaneous HTTP requests from each given Dyno resource.

Web concurrency, in Heroku, therefore, essentially allows each dyno instance to carve up it’s resources to serve multiple concurrent HTTP requests, which it terms WEB_CONCURRENCY. 

Unfortunately this can sometimes lead to Heroku (by default) underestimating resources needed, and running over Dyno memory limits, causing failure, restarts, massive slow-downs due to disk swap having to be used, etc. Basically, you don’t want to have too much web concurrency because it might break your dyno, and the default setting chosen by Heroku may be too high.

You don’t need to worry about this on day one — your app will work. But as you start load testing it, you may find you run into memory overrun issues and all sorts of things like that. If you have a high horsepower python application that chews resources, I suggest you manually set your WEB_CONCURRENCY variable in Heroku command line.
For example:
```
heroku config:set WEB_CONCURRENCY=3
heroku config:set WEB_CONCURRENCY=3 -a <herokuappname>
```

The above statement variations tell Heroku to carve up Dyno resources to support 3 concurrent HTTP requests per Dyno, for all Dynos running a Web process for your app. (So if you have multiple Dynos serving your app in parallel, it automatically sets them all to this same setting).

If performance is not compromised, you can increase web concurrency to increase the number of clients you can serve in parallel, while minimising Dyno cost. If you need to serve more, you can scale Dyno’s horizontally knowing that each one can serve an explicit number of concurrent HTTP requests that you have set.
And of course, you can monitor this with `heroku logs --tail` or in the Heroku dashboard ‘metrics’ section.

## HEROKU PITFALL: Hard limit 30 second request timeout
It’s important to be aware that Heroku has an immutable 30 second timeout for serving HTTP requests. This is a common problem especially encountered by Dash users because many of the data science applications have long load times — see [this post](https://community.plotly.com/t/dash-heroku-timeout/36517).

These might work fine running on your local host, but be aware that your Heroku deployed app must be able to serve within 30 seconds or it will time out. Heroku [docs](https://devcenter.heroku.com/articles/request-timeout) state a few work arounds but take special note of this problem.

## HEROKU PITFALL: Develop on the master/main branch of your GitHub repository
If you’re new to GitHub, just know that you can have multiple ‘branches’ of your project, as you might take it in different directions. These can be merged or left as separate branches. The central branch by default is called ‘master’ or ‘main’ in GitHub. 

When you create your Heroku app it interacts with your GitHub repository to create a kind of Heroku mirror image behind the scenes. If you’re developing your current code on a branch that is not master or main, prepare for pain. It’s not that it can’t be done, I just had a lot of trouble with this when trying to deploy to Heroku. I found the best rule of thumb is to just develop all my code on the default ‘main/master’ branch in my GitHub repository.

## Custom Domain
It’s not too difficult to set up a custom domain for your Heroku app. Obviously, you need to purchase a domain first. Once you’ve done that, the provider will typically have a portal where you can login and adjust settings.

Heroku will generate a unique DNS target in the SETTINGS area of the Heroku web portal dashboard, once logged in. Such as:
```
Animate-salamander-8duwlndghfqbtj0t90uep8bmu.herokudns.com 
```
What you need to do is copy this (your own) DNS target from the Heroku web portal (settings page) and then login to your domain provider portal and for your domain, create a new “CNAME record” with host “www” value “Animate-salamander-8duwlndghfqbtj0t90uep8bmu.herokudns.com” (your unique Heroku DNS target).

If it worked ok, in a few hours your new domain should work!
Essentially all this is doing is redirecting to the Heroku DNS target when someone types your actual domain name. If your new domain is www.blah.com it now has a CNAME record to redirect the incoming HTTP request to Heroku infrastructure, which then serves the actual page (as if you’d typed in blah.herokuapp.com). It’s a tricky thing to set this stuff up because it’s not done very often and you don’t know if it has worked for hours (because it takes time for DNS servers around the world to replicate the new domain list). But there is [good documentation](https://devcenter.heroku.com/articles/custom-domains).

# Flask Caching on Heroku
If you have Flask Caching running on your local machine, it’s straight forward to setup on Heroku with a free Memcachier account. And the docs are good. You can cache to the ephemeral Heroku file-system without Memcachier, noting you might max out your 500MB of Dyno storage, otherwise you can get 100MB free high performance cache via Memcachier.

# Getting Fancy with security and autoscaling etc.
When you want to go to the next level and setup auto-scaling of machines, proper security/authentication etc, I think this is when it starts becoming worth considering Dash Enterprise, upgrading to top-tier Dynos with Heroku (gets $$) OR going down the path of setting up custom infrastructure manually. If you are going manual you could, for example, provision your own virtual machines, set up containerised pipelines using Docker and Kubernetes, manage autoscaling/self healing with Rancher, etc. It’s well and truly DevOps and cloud engineering territory. I’m sure there are lots of other midrange steps you can take like run Docker to build your own containers to deploy, but I want to keep this guide to the bare minimum you need to get on Heroku.

# Closing thoughts
For the newbies and hobbyists out there, I sincerely hope this short novel has been useful to help get your project up and running faster with less pain.
I think Dash is a game-changing tool that is helping to bring data science literacy (and data visualisation technology) to the mainstream public and business world. This can only be a good thing, and this is why I’ve taken the time to write this piece.
