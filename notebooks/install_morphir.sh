# Install node version manager
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# This loads nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  

# morphir requires node >= 16 (would have been great to use apt-get wouldn't it?)
nvm install v16.14.0
npm install -g morphir-elm

# ensure node is installed
NODE_VERSION=`which node`
NPM_VERSION=`which npm`
MORPHIR_VERSION=`which morphir-elm`

if [[ "zz${NODE_VERSION}" == "zz" ]] ; then
  echo "ERROR: could not find node binary"
  exit 1
fi

if [[ "zz${NPM_VERSION}" == "zz" ]] ; then
  echo "ERROR: could not find npm binary"
  exit 1
fi

if [[ "zz${MORPHIR_VERSION}" == "zz" ]] ; then
  echo "ERROR: could not find morphir binary"
  exit 1
fi

# download LCR repo
# appreciate we could do this on a notebook, but we may want to use multiple executors with local install
wget https://github.com/finos/open-reg-tech-us-lcr/zipball/master -O lcr.zip
unzip lcr.zip -d /tmp/lcr
cd /tmp/lcr
mv */* .

# compile morphir LCR project
# sanity check to make sure our infra works
morphir-elm make
if [[ $? -ne 0 ]] ; then
  echo "ERROR: LCR did not compile successfuly"
fi

# make sure to prepend your python OS command with alternative path
#PATH=/root/.nvm/versions/node/v16.14.0/bin

exit 0