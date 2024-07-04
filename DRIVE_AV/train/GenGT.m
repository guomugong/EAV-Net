img_all=dir('label/*.png');

for k=1:length(img_all)
   im=imread(['./label/' img_all(k).name]);
   gt=zeros([size(im,1) size(im,2)]);
   for i=1:size(im,1)
       for j=1:size(im,2)
         if im(i,j,1)== 255 && im(i,j,2) == 0
            gt(i,j)=1;
         end
         if im(i,j,3)== 255 && im(i,j,2) == 0
            gt(i,j)=2;
         end
       end
   end
    imwrite(uint8(gt),['label/' img_all(k).name]);
end