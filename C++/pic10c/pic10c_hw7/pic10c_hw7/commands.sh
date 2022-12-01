mkdir dir1
mkdir dir2
mkdir dir3
touch f1
touch f2
touch f3
rmdir dir3
cp -r dir1 dir3
rmdir dir3
cd dir1
touch foo
cd ..
cp dir1/* dir2
mv dir2/foo dir2/bar
