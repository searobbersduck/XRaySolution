cp -r $1/dr_cls $1/qixiong_cls
sed -i 's|"dr_cls"|"qixiong_cls"|' $1/qixiong_cls/config.pbtxt

