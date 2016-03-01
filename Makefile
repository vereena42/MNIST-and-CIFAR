SUBDIRS = src

VIRTUALENV = env/bin/activate
PYTHON = python2
PYTHON_REQUIREMENTS = requirements.txt
PROMPT = (MNIST&CIFAR)

all: $(SUBDIRS:=_all) $(VIRTUALENV)

%_all:
	$(MAKE) -C $* all

clean: $(SUBDIRS:=_clean)

%_clean:
	$(MAKE) -C $* clean

$(VIRTUALENV): $(PYTHON_REQUIREMENTS)
	virtualenv --python $(PYTHON) --system-site-packages --prompt "$(PROMPT)" env
	. ./env/bin/activate; pip install -r $<

.PHONY: all clean
