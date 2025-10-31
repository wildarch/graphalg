# Jekyll setup

```bash
# See https://jekyllrb.com/docs/installation/ubuntu/
sudo apt-get install ruby-full build-essential zlib1g-dev
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
gem install jekyll bundler
```

Then in the root directory of the site:

```bash
bundle install
bundle exec jekyll serve
```
