journalctl | grep '(firmware)' |tail -n10 |sed -E 's/.* = (.*)s\./\1 /' |sed -E 's/1min/61/' | sort -r | R --slave -e 'x <- scan(file="stdin", quiet=TRUE); summary(x)'
